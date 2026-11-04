"""
Kotodama inference gateway — supervises a fleet of server.py replicas and exposes
a single OpenAI-compatible port that routes/load-balances across them.

What it does:
  * Spawns N replicas (per the config: each (gpu, model) pair gets one server.py;
    a model with a per-spec `gpus:` list is placed only on those GPUs, enabling
    length-aware bucketing — more replicas for verbose models, fewer for terse),
    waits for warmup, polls /health + /memory, and auto-restarts dead/wedged ones.
  * Presents ONE user-facing port. Clients never pick a backend or a model port:
      - POST /v1/chat/completions  -> instruct pool
      - POST /v1/completions       -> base pool
      - the request "model" field overrides the pool when it names a served model
  * Dispatches to the least-loaded healthy replica in the target pool
    (tie-broken by lowest live VRAM), streams responses through transparently,
    and propagates client disconnects to the backend.
  * Admission control: bounded global in-flight -> fast 503 when saturated.

Usage:
    python -m kotodama.serve.gateway --config configs/gateway.example.yaml
    python -m kotodama.serve.gateway --config configs/gateway.example.yaml --gpus 0,1      # subset (testing)

Requires: fastapi, uvicorn, httpx, pyyaml, pydantic
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from kotodama.serve.logs import uvicorn_log_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [gateway] %(message)s")
logger = logging.getLogger("gateway")
# The health loop polls every replica's /health + /memory; silence httpx's
# per-request INFO lines so the request access log stays readable on the console.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Config ────────────────────────────────────────────────────────────────────


class ModelSpec(BaseModel):
    key: str                       # routing pool name, e.g. "base" / "instruct"
    # A serve profile (kotodama.serve.profiles: a name in configs/serve/ or a path): every replica of this pool runs
    # server.py --profile <it>, and the fields below default to the profile's values. One pool is one spec, so all
    # of a pool's replicas run one engine configuration (draws routed to different replicas stay comparable).
    profile: Optional[str] = None
    checkpoint: Optional[str] = None
    mode: Optional[str] = None     # "base" | "chat"
    served_model_name: Optional[str] = None
    model_size: Optional[str] = None
    # Server flags for this pool's replicas only (after the profile and the fleet's serve_args), e.g. the steering
    # banks a pool serves: ["--steer-npz", "<bank>.npz,<partial norms>.npz"].
    args: list[str] = Field(default_factory=list)
    # Optional per-model GPU placement. When set, this model's replicas spawn
    # only on the intersection of this list with the fleet's effective GPUs;
    # when unset (default), the model spawns on EVERY effective GPU (every
    # model x every GPU). This enables length-aware bucketing
    # in ONE gateway: give verbose models (smoltalk ~6x decode time, instruct
    # ~5x) more replicas and terse models fewer, e.g.
    #     - {key: smoltalk, gpus: [0, 1, 2, 3, 4]}
    #     - {key: base,     gpus: [5]}
    # If the intersection with the effective GPUs is EMPTY, the gateway logs a
    # warning and falls back to all effective GPUs rather than silently
    # spawning zero replicas for the pool.
    gpus: Optional[list[int]] = None


class GatewayConfig(BaseModel):
    # The fleet's name: /info reports it, so a client can check what it is talking to before sending traffic.
    name: str = "kotodama-gateway"
    models: list[ModelSpec]
    gpus: list[int] = Field(default_factory=lambda: list(range(8)))
    replica_base_port: int = 2300
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8080

    # Replica launch environment
    python: str = sys.executable
    workdir: str = Field(default_factory=lambda: str(Path.cwd()))
    # Replicas run the server as a module from the installed/PYTHONPATH package.
    serve_module: str = "kotodama.serve.server"
    hf_home: str = Field(default_factory=lambda: os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    log_dir: str = "logs/gateway"
    env_extra: dict[str, str] = Field(default_factory=dict)
    # Extra CLI args appended to every replica's server command
    # (e.g. ["--engine", "fast"]).
    serve_args: list[str] = Field(default_factory=list)

    # Routing / admission
    chat_pool: str = "instruct"            # which pool /v1/chat/completions targets
    completion_pool: str = "base"          # which pool /v1/completions targets
    # Cache-affinity routing: pin a conversation to one replica (rendezvous
    # hash on a stable request-head identity) so server.py's --prefix-cache
    # actually hits across turns. Pure hint: any routing outcome is CORRECT
    # (replicas token-exact-match their cache); affinity only tunes hit rate.
    cache_affinity: bool = True
    # Pin only if the pinned replica has <= this many requests in flight;
    # otherwise fall back to least-busy. 0 = pin only when idle (a cache miss
    # costs one ~40-75ms prefill; queueing behind a long stream costs seconds).
    affinity_max_inflight: int = 0
    max_inflight_global: int = 64          # gateway-wide cap -> 503 over this
    request_timeout_s: float = 900.0       # upstream read timeout (long for streams)
    spawn_stagger_s: float = 0.5           # delay between replica spawns at startup
    # Start one replica per pool first and the rest only once it is ready: with a compile store
    # (server --compile_store / $KOTODAMA_COMPILE_STORE) the first replica compiles and stores the engine's
    # artifacts, the others load them instead of compiling the same graphs at the same time.
    compile_first: bool = False

    # Health / supervision
    health_interval_s: float = 5.0
    warmup_grace_s: float = 300.0          # don't penalize a replica still warming
    unhealthy_threshold: int = 3           # consecutive failed checks -> restart
    restart_backoff_s: float = 15.0        # min seconds between restarts of a replica
    # A replica with failing probes but requests in flight is presumed BUSY, not
    # dead (long batched generations starve /health under saturation), so no
    # unhealthy strikes accrue. But if it ALSO neither passes a probe nor
    # completes a request for this long, treat it as genuinely stuck -> restart.
    stuck_ceiling_s: float = 600.0

    def pool_keys(self) -> set[str]:
        return {m.key for m in self.models}


def load_config(path: Optional[str], gpus_override: Optional[list[int]]) -> GatewayConfig:
    if path:
        raw = yaml.safe_load(Path(path).read_text())
        cfg = GatewayConfig(**raw)
        if not Path(cfg.workdir).is_absolute():
            cfg.workdir = str((Path(path).resolve().parent.parent / cfg.workdir).resolve())
    else:
        raise SystemExit("--config is required (no embedded default fleet)")
    if gpus_override is not None:
        cfg.gpus = gpus_override
    seen: set[str] = set()
    for spec in cfg.models:
        if spec.key in seen:
            raise SystemExit(f"pool '{spec.key}' is configured twice: one pool is one model spec (one profile)")
        seen.add(spec.key)
        _resolve_profile(spec)
    if cfg.compile_first:
        for spec in cfg.models:
            if not compile_store_set(cfg, spec):
                logger.warning(
                    "compile_first is on but pool '%s' sets no compile store (--compile_store in its args or "
                    "serve_args, or KOTODAMA_COMPILE_STORE in env_extra, its profile's env or the gateway's "
                    "environment): its later replicas start from whatever compile cache they share with the first, "
                    "and compile cold if they share none", spec.key)
    # Validate routing targets exist
    keys = cfg.pool_keys()
    for target in (cfg.chat_pool, cfg.completion_pool):
        if target not in keys:
            raise SystemExit(f"routing target '{target}' is not a configured model key {sorted(keys)}")
    return cfg


def compile_store_set(cfg: GatewayConfig, spec: ModelSpec) -> bool:
    """Whether a pool's replicas get a compile store (kotodama.serve.compile_cache), from any place one can be set."""
    if any(a == "--compile_store" or a.startswith("--compile_store=") for a in [*cfg.serve_args, *spec.args]):
        return True
    if cfg.env_extra.get("KOTODAMA_COMPILE_STORE") or os.environ.get("KOTODAMA_COMPILE_STORE"):
        return True
    if spec.profile is not None:
        from kotodama.serve import profiles

        flags, env = profiles.load(spec.profile)
        return bool(flags.get("compile_store") or env.get("KOTODAMA_COMPILE_STORE"))
    return False


_SPEC_FROM_PROFILE = ("checkpoint", "mode", "served_model_name", "model_size")


def _resolve_profile(spec: ModelSpec) -> None:
    """Fill a spec's unset fields from its profile; refuse a spec that still lacks one (every replica needs a
    checkpoint, a mode, a model size and the served name /info is checked against)."""
    if spec.profile is not None:
        from kotodama.serve import profiles

        flags, _env = profiles.load(spec.profile)
        for field in _SPEC_FROM_PROFILE:
            if getattr(spec, field) is None and flags.get(field) is not None:
                setattr(spec, field, str(flags[field]))
    if spec.model_size is None and spec.profile is None:
        spec.model_size = "3b"
    missing = [f for f in _SPEC_FROM_PROFILE if getattr(spec, f) is None]
    if missing:
        raise SystemExit(f"model '{spec.key}' sets no {missing} (directly or through a profile)")


def effective_gpus(cfg: GatewayConfig) -> list[int]:
    """GPUs to actually use. Under Scheduler, CUDA_VISIBLE_DEVICES is the job's
    allocation. The config gpu list is treated as a preference WITHIN that
    allocation (intersection): set it to a subset to pin the fleet away from
    GPUs busy with e.g. training that Scheduler doesn't track, while never
    spawning onto a GPU we weren't granted. Falls back to the full allocation if
    the config names none of the granted GPUs, or to the config list outright on
    a plain manual launch where nothing restricted us."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return cfg.gpus
    allowed = [int(x) for x in cvd.split(",") if x.strip() != ""]
    wanted = [g for g in cfg.gpus if g in allowed]
    chosen = wanted or allowed
    if wanted and len(wanted) < len(allowed):
        wasted = [g for g in allowed if g not in wanted]
        logger.warning(
            "config gpu preference %s shrinks the CUDA_VISIBLE_DEVICES grant %s to %s "
            "— granted-but-unused GPUs %s will sit idle (fleet collapses from %d to %d GPUs); "
            "fix the config's `gpus:` list if this is not a deliberate pin",
            cfg.gpus, allowed, wanted, wasted, len(allowed), len(wanted),
        )
    logger.info("GPU allocation=%s, config prefers=%s -> using %s", allowed, cfg.gpus, chosen)
    return chosen


# ── Replica ─────────────────────────────────────────────────────────────────────


class Replica:
    """One server.py process: a single (gpu, model) backend."""

    def __init__(self, spec: ModelSpec, gpu: int, port: int, log_path: Path) -> None:
        self.spec = spec
        self.gpu = gpu
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None  # None => adopted (not owned)
        self.in_flight = 0
        self.healthy = False
        self.last_vram_gb = 0.0
        self.consecutive_failures = 0
        self.last_activity = 0.0      # last passed probe or completed request
        self.busy_skip_logged = False  # one busy-skip log line per unhealthy episode
        self.started_at = 0.0
        self.last_restart = 0.0
        self.restarts = 0

    @property
    def tag(self) -> str:
        return f"{self.spec.key}@gpu{self.gpu}:{self.port}"

    @property
    def owned(self) -> bool:
        return self.proc is not None

    def status(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "model": self.spec.served_model_name,
            "pool": self.spec.key,
            "gpu": self.gpu,
            "port": self.port,
            "healthy": self.healthy,
            "in_flight": self.in_flight,
            "vram_gb": round(self.last_vram_gb, 2),
            "restarts": self.restarts,
            "owned": self.owned,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else None,
        }


# ── Fleet (supervisor) ───────────────────────────────────────────────────────────


class Fleet:
    def __init__(self, cfg: GatewayConfig) -> None:
        self.cfg = cfg
        self.gpus = effective_gpus(cfg)
        self.replicas: list[Replica] = []
        self.by_pool: dict[str, list[Replica]] = {}
        self.name_to_pool: dict[str, str] = {m.served_model_name: m.key for m in cfg.models}
        # open KV sessions -> the replica holding each (a session lives on one replica; it never moves)
        self.sessions: dict[str, "Replica"] = {}
        self.global_inflight = 0
        self._build()

    def _build(self) -> None:
        log_root = Path(self.cfg.workdir) / self.cfg.log_dir
        log_root.mkdir(parents=True, exist_ok=True)
        # Resolve per-model placement: spec.gpus (when set) intersected with the
        # fleet's effective GPUs; unset -> every effective GPU (legacy behavior).
        placements: list[tuple[ModelSpec, set[int]]] = []
        for spec in self.cfg.models:
            if spec.gpus is None:
                placements.append((spec, set(self.gpus)))
                continue
            subset = [g for g in self.gpus if g in spec.gpus]
            if not subset:
                logger.warning(
                    "model '%s' prefers gpus %s but none are in the effective set %s; "
                    "falling back to ALL effective GPUs so the pool isn't silently empty",
                    spec.key, spec.gpus, self.gpus,
                )
                subset = list(self.gpus)
            placements.append((spec, set(subset)))
        port = self.cfg.replica_base_port
        for gpu in self.gpus:
            for spec, allowed in placements:
                if gpu not in allowed:
                    continue
                log_path = log_root / f"replica_{spec.key}_gpu{gpu}_{port}.log"
                r = Replica(spec, gpu, port, log_path)
                self.replicas.append(r)
                self.by_pool.setdefault(spec.key, []).append(r)
                port += 1

    # -- process lifecycle --

    def spawn(self, r: Replica) -> None:
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(r.gpu),
            "HF_HOME": self.cfg.hf_home,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            **self.cfg.env_extra,
        })
        cmd = [self.cfg.python, "-m", self.cfg.serve_module]
        if r.spec.profile is not None:
            cmd += ["--profile", r.spec.profile]
        cmd += [
            "--checkpoint", r.spec.checkpoint,
            "--model_size", r.spec.model_size,
            "--mode", r.spec.mode,
            "--port", str(r.port),
            "--served_model_name", r.spec.served_model_name,
            "--device", "cuda",
        ]
        cmd.extend(self.cfg.serve_args)
        cmd.extend(r.spec.args)
        logf = open(r.log_path, "ab")
        try:
            r.proc = subprocess.Popen(
                cmd, cwd=self.cfg.workdir, env=env,
                stdout=logf, stderr=logf, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        finally:
            logf.close()  # child keeps its own fd
        r.started_at = time.time()
        r.healthy = False
        r.consecutive_failures = 0
        r.last_restart = time.time()
        logger.info("spawned %s (pid=%s) -> %s", r.tag, r.proc.pid, r.log_path.name)

    def terminate(self, r: Replica) -> None:
        if r.proc is None:
            return
        if r.proc.poll() is None:
            r.proc.terminate()
            try:
                r.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("%s ignored SIGTERM, killing", r.tag)
                r.proc.kill()
                try:
                    r.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        r.proc = None
        r.healthy = False

    def restart(self, r: Replica) -> None:
        if time.time() - r.last_restart < self.cfg.restart_backoff_s:
            return  # back off; let the next health tick try again
        logger.warning("restarting %s (restart #%d)", r.tag, r.restarts + 1)
        self.terminate(r)
        self.spawn(r)
        r.restarts += 1

    def terminate_all(self, grace: float = 10.0) -> None:
        # SIGTERM everything first, then wait in parallel within one grace window
        # (so cancel stays fast even with 16 replicas), SIGKILL stragglers.
        owned = [r for r in self.replicas if r.proc is not None and r.proc.poll() is None]
        for r in owned:
            r.proc.terminate()
        deadline = time.time() + grace
        for r in owned:
            try:
                r.proc.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                logger.warning("%s did not exit within grace; killing", r.tag)
                r.proc.kill()
        for r in self.replicas:
            r.proc = None
            r.healthy = False

    # -- routing --

    def resolve_pool(self, endpoint: str, body: dict[str, Any]) -> str:
        model = body.get("model") if isinstance(body, dict) else None
        if isinstance(model, str) and model:
            if model in self.name_to_pool:
                return self.name_to_pool[model]
            if model in self.by_pool:           # caller passed the pool key directly
                return model
        return self.cfg.chat_pool if endpoint == "chat" else self.cfg.completion_pool

    def pick(self, pool: str, affinity_key: Optional[str] = None) -> tuple[Optional[Replica], str]:
        """Choose a healthy replica. Returns (replica, mode) where mode is
        'pin' (affinity hit), 'lb' (least-busy), 'busy' (last-resort: every
        replica is probe-deaf but alive-and-working), or '-' (no replica)."""
        candidates = [r for r in self.by_pool.get(pool, []) if r.healthy]
        if not candidates:
            # busy≠dead, routing half: a pool whose replicas are ALL decode-deaf
            # to probes still has live workers — queue on the least-busy one
            # rather than 503. Same liveness window as the health checker.
            now = time.time()
            busy_alive = [
                r for r in self.by_pool.get(pool, [])
                if r.in_flight > 0
                and now - max(r.last_activity, r.started_at) < self.cfg.stuck_ceiling_s
            ]
            if busy_alive:
                return min(busy_alive, key=lambda r: (r.in_flight, r.last_vram_gb)), "busy"
            return None, "-"
        if affinity_key and self.cfg.cache_affinity:
            # Rendezvous (highest-random-weight) hash: stable under replica
            # membership changes — a dead replica remaps only its own
            # conversations; everyone else keeps their pin (and their cache).
            pinned = max(
                candidates,
                key=lambda r: hashlib.sha1(f"{affinity_key}|{r.port}".encode()).digest(),
            )
            if pinned.in_flight <= self.cfg.affinity_max_inflight:
                return pinned, "pin"
        # least in-flight, tie-broken by lowest live VRAM
        return min(candidates, key=lambda r: (r.in_flight, r.last_vram_gb)), "lb"

    def acquire(self, r: Replica) -> None:
        r.in_flight += 1
        self.global_inflight += 1

    def release(self, r: Replica) -> None:
        r.in_flight = max(0, r.in_flight - 1)
        self.global_inflight = max(0, self.global_inflight - 1)
        r.last_activity = time.time()  # a completed request proves liveness


# ── Globals ───────────────────────────────────────────────────────────────────

_cfg: GatewayConfig
_fleet: Fleet
_http: httpx.AsyncClient
_health_task: Optional[asyncio.Task] = None


# ── Health / supervision loop ────────────────────────────────────────────────────


async def _check_replica(r: Replica) -> None:
    # Detect an owned process that exited -> respawn immediately.
    if r.owned and r.proc is not None and r.proc.poll() is not None:
        logger.warning("%s process exited (rc=%s); respawning", r.tag, r.proc.returncode)
        r.healthy = False
        _fleet.spawn(r)
        r.restarts += 1
        return

    try:
        resp = await _http.get(f"{r.url}/health", timeout=3.0)
        ok = resp.status_code == 200 and bool(resp.json().get("model_loaded"))
    except Exception:
        ok = False

    if ok:
        if not r.healthy:
            logger.info("%s is healthy", r.tag)
        r.healthy = True
        r.consecutive_failures = 0
        r.last_activity = time.time()
        r.busy_skip_logged = False
        try:
            mresp = await _http.get(f"{r.url}/memory", timeout=3.0)
            r.last_vram_gb = float(mresp.json().get("allocated_gb", 0.0))
        except Exception:
            pass
        return

    # Not OK. Tolerate the warmup window (port isn't bound until warmup completes).
    r.healthy = False
    still_warming = r.started_at and (time.time() - r.started_at) < _cfg.warmup_grace_s
    if still_warming:
        return
    if r.in_flight > 0:
        # Failing probes with requests in flight = presumed BUSY, not dead
        # (long batched generations starve /health under saturation). Don't
        # strike — unless nothing has passed a probe or completed for
        # stuck_ceiling_s, in which case it's genuinely wedged.
        last_alive = max(r.last_activity, r.started_at)
        if time.time() - last_alive < _cfg.stuck_ceiling_s:
            if not r.busy_skip_logged:
                logger.info("%s probe failed with %d in flight; presuming busy, skipping strikes", r.tag, r.in_flight)
                r.busy_skip_logged = True
            return
        logger.warning("%s stuck: probes failing, %d in flight, no activity for %.0fs; restarting", r.tag, r.in_flight, time.time() - last_alive)
        _fleet.restart(r)
        return
    r.consecutive_failures += 1
    if r.consecutive_failures >= _cfg.unhealthy_threshold:
        logger.warning("%s failed %d checks past warmup; restarting", r.tag, r.consecutive_failures)
        _fleet.restart(r)


async def health_loop() -> None:
    while True:
        try:
            await asyncio.gather(*(_check_replica(r) for r in _fleet.replicas))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("health loop iteration failed")
        await asyncio.sleep(_cfg.health_interval_s)


async def _wait_ready(r: Replica) -> bool:
    """Poll one replica until it reports its model loaded, its process exits, or the warmup grace runs out."""
    deadline = time.time() + _cfg.warmup_grace_s
    while time.time() < deadline:
        if r.proc is not None and r.proc.poll() is not None:
            logger.warning("%s exited during startup (code %s)", r.tag, r.proc.returncode)
            return False
        try:
            resp = await _http.get(f"{r.url}/health", timeout=2.0)
            if resp.status_code == 200 and resp.json().get("model_loaded"):
                r.healthy = True
                r.last_activity = time.time()
                return True
        except Exception:
            pass
        await asyncio.sleep(_cfg.health_interval_s)
    return False


async def _adopt_or_spawn() -> None:
    """Reuse an already-healthy replica on the expected port (daemon restart),
    otherwise spawn a fresh one. Staggered to avoid a thundering herd; with
    ``compile_first``, each pool's first replica is ready before the rest start."""
    first_of_pool: set[str] = set()
    for r in _fleet.replicas:
        adopted = False
        try:
            resp = await _http.get(f"{r.url}/health", timeout=2.0)
            if resp.status_code == 200 and resp.json().get("model_loaded"):
                # Confirm it's actually our model before adopting the port.
                info = await _http.get(f"{r.url}/info", timeout=2.0)
                if info.json().get("name") == r.spec.served_model_name:
                    r.healthy = True
                    r.started_at = time.time()
                    adopted = True
                    logger.info("adopted existing healthy replica %s", r.tag)
        except Exception:
            adopted = False
        if not adopted:
            _fleet.spawn(r)
            if _cfg.compile_first and r.spec.key not in first_of_pool and len(_fleet.by_pool[r.spec.key]) > 1:
                logger.info("compile_first: waiting for %s before starting the rest of pool '%s'", r.tag, r.spec.key)
                t0 = time.time()
                ok = await _wait_ready(r)
                logger.info("compile_first: %s %s after %.0fs", r.tag, "ready" if ok else "NOT ready", time.time() - t0)
            await asyncio.sleep(_cfg.spawn_stagger_s)
        first_of_pool.add(r.spec.key)


# ── App ───────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http, _health_task
    _http = httpx.AsyncClient(timeout=httpx.Timeout(_cfg.request_timeout_s, connect=5.0))
    logger.info("fleet: %d replicas across gpus %s (%d models)",
                len(_fleet.replicas), _fleet.gpus, len(_cfg.models))
    await _adopt_or_spawn()
    _health_task = asyncio.create_task(health_loop())
    try:
        yield
    finally:
        if _health_task is not None:
            _health_task.cancel()
            try:
                await _health_task
            except asyncio.CancelledError:
                pass
        logger.info("shutting down: terminating owned replicas")
        _fleet.terminate_all()
        await _http.aclose()


app = FastAPI(title="kotodama gateway", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _overloaded() -> bool:
    return _fleet.global_inflight >= _cfg.max_inflight_global


def _affinity_key(body: dict[str, Any]) -> Optional[str]:
    """Stable conversation identity for cache-affinity routing.

    Multi-turn requests grow by APPENDING (that is what makes them cacheable),
    so the HEAD of the request is a stable identity across a conversation's
    turns while still distinguishing different conversations:
      * explicit OpenAI `user` field, if the client sets one (explicit wins);
      * chat: the system message head plus the FIRST user message head
        (system alone would collide every conversation sharing a template);
      * completion/native: the first 256 chars of the prompt (a token-id prompt: its first 64 ids).
    Returns None when nothing usable exists (-> plain least-busy routing).
    """
    user = body.get("user")
    if isinstance(user, str) and user:
        return f"user:{user}"
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        head = f"{msgs[0].get('role', '')}:{str(msgs[0].get('content', ''))[:256]}"
        if (
            msgs[0].get("role") == "system"
            and len(msgs) > 1
            and isinstance(msgs[1], dict)
        ):
            head += f"|{msgs[1].get('role', '')}:{str(msgs[1].get('content', ''))[:256]}"
        return head
    prompt = body.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt[:256]
    ids = body.get("prompt_ids", prompt)
    if isinstance(ids, list) and ids and all(isinstance(t, int) for t in ids[:64]):
        return "ids:" + ",".join(str(t) for t in ids[:64])  # a token-id prompt: its first 64 ids
    return None


async def _proxy(request: Request, endpoint: str, path: str) -> Response:
    t0 = time.monotonic()
    try:
        body = await request.json()
    except Exception:
        logger.warning("400 %s | invalid JSON body", path)
        return JSONResponse(status_code=400, content={"error": "invalid JSON body"})

    pool = _fleet.resolve_pool(endpoint, body)
    if not path.startswith("/v1/") and isinstance(body, dict) and "model" in body:
        # the native endpoints take no `model` field: the gateway routes on it here and forwards the rest; a name no
        # pool serves is refused rather than answered by the default pool
        model = body.pop("model")
        if model is not None and model not in _fleet.name_to_pool and model not in _fleet.by_pool:
            logger.warning("400 %s | model %r is not served by this fleet", path, model)
            return JSONResponse(status_code=400, content={"error": f"model {model!r} is not served by this fleet; "
                                                                   f"served: {sorted(_fleet.name_to_pool)}"})
    if _overloaded():
        logger.warning("503 %s pool=%s | overloaded (inflight=%d)", path, pool, _fleet.global_inflight)
        return JSONResponse(status_code=503, content={"error": "gateway overloaded, retry shortly"})

    sid = body.get("kv_session") if isinstance(body, dict) else None
    akey = _affinity_key(body)
    if sid is not None:  # a KV session's turn goes to the replica holding the session, or nowhere
        replica, err = _session_replica(str(sid))
        if replica is None:
            return err
        route_mode = "session"
    else:
        replica, route_mode = _fleet.pick(pool, akey)
    if replica is None:
        logger.warning("503 %s pool=%s | no healthy replica", path, pool)
        return JSONResponse(status_code=503, content={"error": f"no healthy '{pool}' replica available"})

    stream = bool(body.get("stream"))
    _fleet.acquire(replica)

    if stream:
        logger.info("%s -> %s [stream:%s] (inflight=%d)", path, replica.tag, route_mode, _fleet.global_inflight)
        # Hand the released-on-exit generator to StreamingResponse.
        return StreamingResponse(
            _stream_upstream(replica, path, body, t0),
            media_type="text/event-stream",
        )

    # Non-streaming: forward, with a single failover to another replica on a
    # transport error (the first may have just died between pick and send).
    try:
        try:
            resp = await _http.post(f"{replica.url}{path}", json=body)
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
            logger.warning("%s transport error (%s); marking unhealthy + failover", replica.tag, exc)
            replica.healthy = False
            if sid is not None:  # a session's KV is on that replica: no other replica can serve its turn
                return JSONResponse(status_code=502, content={"error": f"backend error on the replica holding "
                                                                       f"session {sid!r}: {exc}"})
            _fleet.release(replica)
            # Re-pick with the same key: rendezvous skips the now-unhealthy
            # replica and gives this conversation a stable new home.
            alt, route_mode = _fleet.pick(pool, akey)
            if alt is None:
                logger.warning("502 %s pool=%s | no replica for failover", path, pool)
                return JSONResponse(status_code=502, content={"error": "backend error, no replica for failover"})
            replica = alt
            _fleet.acquire(replica)
            resp = await _http.post(f"{replica.url}{path}", json=body)
        logger.info("%s -> %s [%s] | %d | %.0fms", path, replica.tag, route_mode, resp.status_code, (time.monotonic() - t0) * 1000)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except Exception as exc:
        logger.warning("502 %s -> %s | %s", path, replica.tag, exc)
        return JSONResponse(status_code=502, content={"error": f"backend error: {exc}"})
    finally:
        _fleet.release(replica)


async def _stream_upstream(replica: Replica, path: str, body: dict[str, Any], t0: float):
    """Proxy an SSE stream. On downstream disconnect, StreamingResponse throws
    GeneratorExit here; exiting the `async with` closes the upstream connection,
    which trips server.py's is_disconnected() so the backend stops + frees."""
    outcome = "done"
    try:
        async with _http.stream("POST", f"{replica.url}{path}", json=body) as upstream:
            async for chunk in upstream.aiter_raw():
                yield chunk
    except GeneratorExit:
        outcome = "client-disconnect"
        raise
    except Exception as exc:
        outcome = "backend-error"
        logger.warning("%s stream error: %s", replica.tag, exc)
        yield b'data: {"error": "backend stream error"}\n\n'
    finally:
        logger.info("%s -> %s [stream:%s] | %.0fms", path, replica.tag, outcome, (time.monotonic() - t0) * 1000)
        _fleet.release(replica)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _proxy(request, endpoint="chat", path="/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _proxy(request, endpoint="completion", path="/v1/completions")


@app.post("/generate")
async def generate(request: Request):
    # Native passthrough. No messages-vs-prompt knowledge here, so default to the
    # completion pool unless the body's "model" field says otherwise.
    return await _proxy(request, endpoint="completion", path="/generate")


@app.post("/score_tokens")
async def score_tokens(request: Request):
    return await _proxy(request, endpoint="completion", path="/score_tokens")


@app.post("/logprobs")
async def logprobs(request: Request):
    return await _proxy(request, endpoint="completion", path="/logprobs")


def _session_replica(sid: str) -> tuple[Optional["Replica"], Optional[JSONResponse]]:
    replica = _fleet.sessions.get(sid)
    if replica is None:
        return None, JSONResponse(status_code=404, content={"error": f"no open KV session {sid!r} on this fleet"})
    if not replica.healthy:
        return None, JSONResponse(status_code=410, content={"error": f"the replica holding KV session {sid!r} is "
                                                                     "down; the session is lost"})
    return replica, None


async def _forward_session(replica: "Replica", method: str, path: str, **kw: Any) -> Response:
    _fleet.acquire(replica)
    try:
        resp = await _http.request(method, f"{replica.url}{path}", **kw)
    except Exception as exc:
        logger.warning("502 %s -> %s | %s", path, replica.tag, exc)
        return JSONResponse(status_code=502, content={"error": f"backend error: {exc}"})
    finally:
        _fleet.release(replica)
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"))


@app.post("/kv_session/open")
async def kv_session_open(request: Request):
    """Open a KV session on one replica of the pool the body's ``model`` names (else the completion pool), the
    least loaded; every later request of the session goes to that replica."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    pool = _fleet.resolve_pool("completion", body if isinstance(body, dict) else {})
    replica, _ = _fleet.pick(pool, None)
    if replica is None:
        return JSONResponse(status_code=503, content={"error": f"no healthy '{pool}' replica available"})
    payload = {k: v for k, v in (body or {}).items() if k != "model"} if isinstance(body, dict) else {}
    resp = await _forward_session(replica, "POST", "/kv_session/open", json=payload)
    if resp.status_code == 200:
        sid = json.loads(resp.body)["session"]
        _fleet.sessions[sid] = replica
        logger.info("kv session %s opened on %s", sid, replica.tag)
    return resp


@app.post("/kv_session/{op}")
async def kv_session_op(op: str, request: Request):
    """close, append, evict, export, import: forwarded to the replica holding the session (``session`` in the JSON
    body, or in the query for import, whose body is an export's bytes)."""
    if op not in ("close", "append", "evict", "export", "import"):
        return JSONResponse(status_code=404, content={"error": f"no KV session op {op!r}"})
    if op == "import":
        sid = request.query_params.get("session", "")
        kw: dict[str, Any] = {"content": await request.body(), "params": dict(request.query_params),
                              "headers": {"content-type": "application/octet-stream"}}
    else:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "invalid JSON body"})
        sid = str(body.get("session", "")) if isinstance(body, dict) else ""
        kw = {"json": body}
    replica, err = _session_replica(sid)
    if replica is None:
        return err
    resp = await _forward_session(replica, "POST", f"/kv_session/{op}", **kw)
    if op == "close" and resp.status_code in (200, 404):
        _fleet.sessions.pop(sid, None)
    return resp


@app.get("/kv_session/{sid}")
async def kv_session_info(sid: str):
    replica, err = _session_replica(sid)
    if replica is None:
        return err
    return await _forward_session(replica, "GET", f"/kv_session/{sid}")


@app.get("/info")
async def info():
    """The fleet: its name, and per pool the served model, its profile and how many replicas are ready."""
    return {
        "name": _cfg.name,
        "gateway": True,
        "pools": {
            m.key: {"model": m.served_model_name, "profile": m.profile, "mode": m.mode,
                    "replicas_ready": sum(r.healthy for r in _fleet.by_pool.get(m.key, [])),
                    "replicas_total": len(_fleet.by_pool.get(m.key, []))}
            for m in _cfg.models
        },
        "chat_pool": _cfg.chat_pool,
        "completion_pool": _cfg.completion_pool,
    }


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": m.served_model_name, "object": "model", "owned_by": "aethera-gp"}
            for m in _cfg.models
        ],
    }


@app.get("/health")
async def health():
    ready = sum(r.healthy for r in _fleet.replicas)
    total = len(_fleet.replicas)
    by_pool = {
        pool: sum(r.healthy for r in reps)
        for pool, reps in _fleet.by_pool.items()
    }
    return {
        "status": "ok" if ready == total else ("degraded" if ready else "down"),
        "replicas_ready": ready,
        "replicas_total": total,
        "ready_by_pool": by_pool,
        "global_inflight": _fleet.global_inflight,
    }


@app.get("/status")
async def status():
    return {
        "gpus": _fleet.gpus,
        "global_inflight": _fleet.global_inflight,
        "max_inflight_global": _cfg.max_inflight_global,
        "replicas": [r.status() for r in _fleet.replicas],
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kotodama inference gateway")
    parser.add_argument("--config", required=True, help="Path to gateway YAML config")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU override (e.g. 0,1) for testing on a subset")
    parser.add_argument("--host", default=None, help="Override gateway host")
    parser.add_argument("--port", type=int, default=None, help="Override gateway port")
    args = parser.parse_args()

    gpus_override = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    _cfg = load_config(args.config, gpus_override)
    if args.host:
        _cfg.gateway_host = args.host
    if args.port:
        _cfg.gateway_port = args.port
    _fleet = Fleet(_cfg)

    uvicorn.run(app, host=_cfg.gateway_host, port=_cfg.gateway_port, log_level="info",
                log_config=uvicorn_log_config())
