'''transcript_sandbox.py  (voice mode, Deepgram SDK v4, sync)
================================================================
Interactive sandbox for **TranscriptManager**.

All shared voice helpers now live in `utils.py` to avoid duplication
with the task‑list sandbox.
'''

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unify
from constants import LOGGER as _LG
from communication.transcript_manager.transcript_manager import TranscriptManager
from communication.types.message import Message

# Voice helpers (PortAudio capture, Deepgram STT, Cartesia TTS)
from utils import record_until_enter, transcribe_deepgram, speak

# ---------------------------------------------------------------------------
# Scenario seeding data
# ---------------------------------------------------------------------------

_CONTACTS: List[dict] = [
    dict(first_name='Carlos', surname='Diaz', email_address='carlos.diaz@example.com', phone_number='+14155550000', whatsapp_number='+14155550000'),
    dict(first_name='Dan', surname='Turner', email_address='dan.turner@example.com', phone_number='+447700900001', whatsapp_number='+447700900001'),
    dict(first_name='Julia', surname='Nguyen', email_address='julia.nguyen@example.com', phone_number='+447700900002', whatsapp_number='+447700900002'),
    dict(first_name='Jimmy', surname="O'Brien", email_address='jimmy.obrien@example.com', phone_number='+61240011000', whatsapp_number='+61240011000'),
    dict(first_name='Anne', surname='Fischer', email_address='anne.fischer@example.com', phone_number='+49891234567', whatsapp_number='+49891234567'),
    dict(first_name='Leo', surname='Kowalski', email_address='leo.k@example.com', phone_number='+48500100200', whatsapp_number='+48500100200'),
    dict(first_name='Sara', surname='Jensen', email_address='sara.j@example.com', phone_number='+46700111222', whatsapp_number='+46700111222'),
]

_ID_BY_NAME: Dict[str, int] = {}


def _seed_fixed(tm: TranscriptManager) -> None:
    'Populate contacts and a rich set of exchanges.'
    for idx, c in enumerate(_CONTACTS):
        tm.create_contact(**c)
        _ID_BY_NAME[c['first_name'].lower()] = idx

    base = datetime(2025, 4, 20, 9, 0, tzinfo=timezone.utc)
    ex_id = 0

    def _cid(name: str) -> int:
        return _ID_BY_NAME[name]

    # Dan ↔ Julia phone calls
    tm.log_messages([
        Message(medium='phone_call', sender_id=_cid('dan'), receiver_id=_cid('julia'), timestamp=base.isoformat(), content='Hi Julia, it\'s Dan. Quick check‑in about Q2 metrics.', exchange_id=ex_id),
        Message(medium='phone_call', sender_id=_cid('julia'), receiver_id=_cid('dan'), timestamp=(base + timedelta(seconds=30)).isoformat(), content='Sure Dan, ready when you are.', exchange_id=ex_id),
    ])
    ex_id += 1

    later = base + timedelta(days=6, minutes=30)
    tm.log_messages([
        Message(medium='phone_call', sender_id=_cid('dan'), receiver_id=_cid('julia'), timestamp=later.isoformat(), content='Morning Julia – finalising the London event agenda today.', exchange_id=ex_id),
        Message(medium='phone_call', sender_id=_cid('julia'), receiver_id=_cid('dan'), timestamp=(later + timedelta(seconds=45)).isoformat(), content='Great. Let\'s confirm the speaker list and coffee budget.', exchange_id=ex_id),
    ])
    ex_id += 1

    # Carlos bulk‑order email chain
    t_email = base + timedelta(days=1, hours=3)
    tm.log_messages([
        Message(medium='email', sender_id=_cid('carlos'), receiver_id=_cid('dan'), timestamp=t_email.isoformat(), content='Subject: Stapler bulk order\n\nHi Dan, I\'m **interested in buying 200 units** of your new stapler. Can you quote?\n\nThanks,\nCarlos', exchange_id=ex_id),
        Message(medium='email', sender_id=_cid('dan'), receiver_id=_cid('carlos'), timestamp=(t_email + timedelta(hours=2)).isoformat(), content='Hi Carlos — sure, $4.50 per unit. See attached PDF.', exchange_id=ex_id),
    ])
    ex_id += 1

    # Jimmy holiday heads‑up
    t_holiday = base + timedelta(days=2, hours=9, minutes=10)
    tm.log_messages([
        Message(medium='whatsapp_message', sender_id=_cid('jimmy'), receiver_id=_cid('dan'), timestamp=t_holiday.isoformat(), content='Heads‑up Dan, I\'ll be **on holiday from 2025‑05‑15 to 2025‑05‑30**. Ping me before that if urgent.', exchange_id=ex_id),
    ])
    ex_id += 1

    # Anne passport excuse
    t_excuse = base + timedelta(days=3)
    tm.log_messages([
        Message(medium='whatsapp_message', sender_id=_cid('anne'), receiver_id=_cid('dan'), timestamp=t_excuse.isoformat(), content='Sorry Dan, I can\'t join the Berlin trip because my passport expired last week.', exchange_id=ex_id),
    ])
    ex_id += 1

    # Leo logistics SMS
    t_log = base + timedelta(days=4, hours=13)
    tm.log_messages([
        Message(medium='sms_message', sender_id=_cid('leo'), receiver_id=_cid('dan'), timestamp=t_log.isoformat(), content='Dan, the pallets are delayed at customs. Need new clearance docs by tomorrow.', exchange_id=ex_id),
        Message(medium='sms_message', sender_id=_cid('dan'), receiver_id=_cid('leo'), timestamp=(t_log + timedelta(minutes=7)).isoformat(), content='On it – forwarding the updated invoices now.', exchange_id=ex_id),
    ])
    ex_id += 1

    # Sara voice‑note (text representation)
    t_voice = base + timedelta(days=5, hours=17)
    tm.log_messages([
        Message(medium='whatsapp_message', sender_id=_cid('sara'), receiver_id=_cid('dan'), timestamp=t_voice.isoformat(), content='(voice) Hey Dan! Quick thing – keynote speaker wants to swap slots with the panel. OK?', exchange_id=ex_id),
    ])
    ex_id += 1

    # Filler exchanges for noise
    random.seed(42)
    media = ['email', 'phone_call', 'sms_message', 'whatsapp_message']
    filler_start = base + timedelta(days=6)
    for extra in range(25):
        mid = random.choice(media)
        a, b = random.sample(list(_ID_BY_NAME.values()), 2)
        ex_ref = ex_id + extra
        msgs: List[Message] = []
        for i in range(random.randint(5, 15)):
            msgs.append(Message(medium=mid, sender_id=a if i % 2 else b, receiver_id=b if i % 2 else a, timestamp=(filler_start + timedelta(minutes=extra * 4 + i)).isoformat(), content=f'Filler {ex_ref}-{i} {mid} banter.', exchange_id=ex_ref))
        tm.log_messages(msgs)

    tm.summarize(exchange_ids=[0, 1])


def _seed_llm(tm: TranscriptManager) -> Optional[str]:
    prompt = (
        'Create a realistic communication history for a European event agency with 8‑12 contacts interacting via email, phone_call, sms_message and whatsapp_message. ' \
        'Include 60‑90 exchanges with 3‑8 messages each, timestamps between 2025‑04‑01 and 2025‑05‑10 (UTC). ' \
        'Return JSON with keys: contacts, exchanges, theme.'
    )

    client = unify.Unify('o4-mini@openai', cache=True)
    client.set_system_message(prompt)
    raw = client.generate('Produce scenario').strip()

    try:
        payload = json.loads(raw)
    except Exception:
        _LG.warning('LLM scenario failed – using fixed seed.')
        _seed_fixed(tm)
        return None

    for idx, c in enumerate(payload['contacts']):
        tm.create_contact(**c)
        _ID_BY_NAME[c['first_name'].lower()] = idx

    for ex_id, ex in enumerate(payload['exchanges']):
        mid = ex.get('medium', 'email')
        msgs = [
            Message(medium=mid, sender_id=_ID_BY_NAME[m['sender_first_name'].lower()], receiver_id=_ID_BY_NAME[m['receiver_first_name'].lower()], timestamp=m['timestamp'], content=m['content'], exchange_id=ex_id)
            for m in ex['messages']
        ]
        tm.log_messages(msgs)

    if summaries := [e.get('summary') for e in payload['exchanges'] if e.get('summary')]:
        tm.summarize(texts=summaries)

    return payload.get('theme')


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _dispatch(tm: TranscriptManager, raw: str, *, show_steps: bool) -> Tuple[str, str, List | None]:
    ans, steps = tm.ask(raw.strip(), return_reasoning_steps=show_steps)
    return 'ask', ans, steps


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='TranscriptManager sandbox with Deepgram voice mode')
    parser.add_argument('--voice', '-v', action='store_true', help='enable voice capture/playback')
    parser.add_argument('--scenario', choices=['fixed', 'llm'], default='fixed')
    parser.add_argument('--new', '-n', action='store_true', help='wipe & reseed data')
    parser.add_argument('--silent', '-s', action='store_true', help='suppress tool logs')
    parser.add_argument('--debug', '-d', action='store_true', help='verbose HTTP/LLM logging')
    args = parser.parse_args()

    if not args.silent:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        _LG.setLevel(logging.INFO)
        if not args.debug:
            for noisy in ('unify', 'unify.utils', 'unify.logging', 'requests', 'httpx'):
                logging.get
