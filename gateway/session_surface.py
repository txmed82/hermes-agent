"""Per-turn client surface (floating HUD, live voice, graphical app) carried through canonical admission.

A surface is a fact about ONE turn, never about the session: the same session alternates
between the app window, the HUD and a spoken delegation. It is committed with the admission
row and reaches the model as a note on the user message (api_content sidecar), so the
persisted user row stays the words the user typed and the frozen system prefix is untouched.

``app`` is the renderer fact for a graphical client (Pantheon) resuming a session whose creation
provenance is a terminal (``cli``/``tui``): the agent platform, its system hint and its tool scope stay
the terminal's, and only that turn learns its reply is rendered by an app that turns a
``MEDIA:/absolute/path`` line into a download card. Creation origin is not the renderer.
"""
from contextlib import contextmanager
from contextvars import ContextVar

from hermes_state_runtime import RuntimeStoreError

SURFACES = frozenset({'hud', 'voice-live', 'app'})
# Agent platforms whose system hint says there is no attachment channel; only they need the app note.
_TERMINAL_PLATFORMS = frozenset({'cli', 'tui'})
_VOICE_CONTEXT_LIMIT = 6000
_SUBMIT_FIELDS = ('surface', 'voice_context', 'interrupted', 'voice_turn')
_surface_turn = ContextVar('surface_turn', default=None)


def submit_surface_fields(params):
    """The raw per-turn surface fields of a ``prompt.submit`` request, forwarded to admission."""
    return {key: params[key] for key in _SUBMIT_FIELDS if key in params}


def admit_surface(params):
    """Wire ``surface`` / ``voice_context`` / ``interrupted`` / ``voice_turn`` -> committed ``surface_v1``
    (``{}`` when absent). ``voice_turn`` marks a spoken voice-mode turn: it runs on ``auxiliary.voice_chat``."""
    surface, context, interrupted, voice_turn = (params.get(key) for key in _SUBMIT_FIELDS)
    if surface is not None and surface not in SURFACES:
        raise RuntimeStoreError('invalid_params')
    # The spoken transcript only makes sense for a live-voice delegation; anywhere else it is
    # a client smuggling model input past the persisted row.
    if context is not None and (surface != 'voice-live' or not isinstance(context, str)):
        raise RuntimeStoreError('invalid_params')
    if any(flag is not None and type(flag) is not bool for flag in (interrupted, voice_turn)):
        raise RuntimeStoreError('invalid_params')
    committed = {}
    if surface:
        committed['surface'] = surface
    if context:
        committed['voice_context'] = context[:_VOICE_CONTEXT_LIMIT]
    if interrupted:
        committed['interrupted'] = True
    if voice_turn:
        committed['voice_turn'] = True
    return {'surface_v1': committed} if committed else {}


def restore_surface(committed):
    """A committed ``surface_v1`` arriving across a process boundary (managed worker), re-checked by
    the admission validator: only an object admission itself could have committed is accepted."""
    if (not isinstance(committed, dict) or not committed or set(committed) - set(_SUBMIT_FIELDS)
            or admit_surface(committed).get('surface_v1', {}) != committed):
        raise RuntimeStoreError('invalid_params')
    return committed


def _hud_note(committed, valid_tool_names):
    from agent.prompt_builder import hud_surface_note
    return hud_surface_note(valid_tool_names)


def _voice_live_note(committed, valid_tool_names):
    from tools.voice_live import voice_live_turn_note
    return voice_live_turn_note(committed.get('voice_context') or '')


APP_SURFACE_NOTE = (
    "[Note: this message was typed in a graphical Hermes app that renders your reply as Markdown, not in a "
    "terminal. For this turn the terminal file guidance does not apply: to hand the user a file that exists on "
    "this machine, put MEDIA:/absolute/path/to/file alone on its own line (no spaces in the path, not inside a "
    "code block); the app shows it as a tap-to-download card, and image files render inline.]"
)


def _app_note(committed, valid_tool_names, platform=None):
    # Only a terminal-platform agent was told there is no attachment channel; a desktop or messaging
    # agent already has its renderer's own contract in the system hint.
    return APP_SURFACE_NOTE if platform in _TERMINAL_PLATFORMS else ''


_SURFACE_NOTES = {'hud': _hud_note, 'voice-live': _voice_live_note}


def surface_note(committed, valid_tool_names=None, platform=None):
    """The model-bound note for a committed surface (``""`` for the plain app window): barge-in
    first, then the HUD read-the-window-below prior, the spoken-delegation contract, or the
    graphical-app file-delivery contract (terminal-platform agents only)."""
    notes = []
    if committed.get('interrupted'):
        from tools.tts_streaming import SPEECH_INTERRUPTED_NOTE
        notes.append(SPEECH_INTERRUPTED_NOTE)
    if committed.get('surface') == 'app':
        notes.append(_app_note(committed, valid_tool_names, platform))
    render = _SURFACE_NOTES.get(committed.get('surface'))
    if render is not None:
        notes.append(render(committed, valid_tool_names))
    return '\n\n'.join(note for note in notes if note)


@contextmanager
def surface_turn_scope(committed):
    token = _surface_turn.set(committed)
    try:
        yield
    finally:
        _surface_turn.reset(token)


def surface_voice_turn():
    """Whether the executing admission is a spoken voice turn (``auxiliary.voice_chat`` route)."""
    return bool((_surface_turn.get() or {}).get('voice_turn'))


def surface_turn_note(agent):
    """The executing admission's surface note, gated on the tools this agent actually has."""
    committed = _surface_turn.get()
    if not committed:
        return ''
    return surface_note(committed, getattr(agent, 'valid_tool_names', None), getattr(agent, 'platform', None))


def arm_surface_turn(agent, notes=(), voice_turn=False):
    """Project the executing admission's surface onto the agent's one-turn inputs, the same way in
    process and in a managed worker: the ``auxiliary.voice_chat`` marker and the must-deliver notes
    that ride the current user message (never the system prompt). Assigned unconditionally so a
    reused agent never replays a stale note or route."""
    agent._voice_turn_pending = bool(voice_turn) or surface_voice_turn()
    agent._gateway_turn_context_notes = '\n\n'.join(
        note for note in (*notes, surface_turn_note(agent)) if note)
