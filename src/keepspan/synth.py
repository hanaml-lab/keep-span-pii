"""Presidio-style synthetic documents: hand-written templates filled with Faker values.

Every value Faker fills in is a span to redact, with its type. Filler that is not personal data (an order number,
a product, a job title, an amount) is drawn from Faker too but carries no span, so the documents contain numbers
and capitalised words that must be left alone. Generation is deterministic for a given seed.
"""

from __future__ import annotations

import random
import re

from faker import Faker

LOCALES = (("en_US", 0.5), ("en_GB", 0.2), ("en_CA", 0.1), ("en_AU", 0.1), ("en_IN", 0.1))

# Placeholder -> span type (None: filler, no span).
SLOTS: dict[str, str | None] = {
    "name": "NAME", "first": "NAME", "last": "NAME", "name2": "NAME",
    "email": "CONTACT", "phone": "CONTACT",
    "street": "LOC", "city": "LOC", "zip": "LOC",
    "dob": "DATE", "date": "DATE",
    "ssn": "ID", "passport": "ID", "dl": "ID", "card": "ID", "iban": "ID", "account": "ID", "ip": "ID",
    "age": "QUASI",
    "order": None, "product": None, "job": None, "amount": None, "weekday": None, "dept": None,
}

TEMPLATES: tuple[str, ...] = (
    "Hi {first}, your order {order} for the {product} has shipped to {street}, {city} {zip}.",
    "Please update my phone number to {phone}. Thanks, {name}",
    "Patient: {name}\nDOB: {dob}\nAddress: {street}, {city}, {zip}\nPhone: {phone}",
    "I can't log in. My account email is {email} and the last card I used ends in {card}.",
    "Dear {name},\n\nWe received your application for the {job} position on {date}. "
    "We will contact you at {email} within two weeks.\n\nRegards,\nHiring team",
    "Can you remind {first} that the {dept} review moved to {weekday}? Her cell is {phone}.",
    "Wire {amount} to IBAN {iban}, reference {order}, before {date}.",
    "My SSN is {ssn} and my date of birth is {dob}. Why was my claim rejected?",
    "{name} ({age}) was admitted on {date} and discharged two days later.",
    "Shipping label\n{name}\n{street}\n{city} {zip}",
    "New login to your account from IP {ip} on {date}. If this wasn't you, reply to this message.",
    "Passport number {passport}, issued to {name}, expires {date}.",
    "Driver licence {dl} belongs to {name}, who lives at {street} in {city}.",
    "Hey, it's {first}. Can you send the {product} invoice to {email}? Order {order}.",
    "Customer {name} called about account {account}; callback number {phone}.",
    "Refund of {amount} issued to card {card} for order {order}.",
    "{first} {last} moved from {city} to {street}, {zip} last month.",
    "Meeting notes: {name} will own the {dept} budget; {name2} takes the {product} launch.",
    "Emergency contact: {name2}, {phone}, relationship: sister.",
    "Your appointment with Dr {last} is confirmed for {date}. Call {phone} to reschedule.",
    "The tenant, {name}, aged {age}, has paid rent for {street}, {city} since {date}.",
    "Forwarding address for {name}: {street}, {city}, {zip}. Email: {email}",
    "I'm {first}, {age}, working as a {job}. You can reach me at {phone} or {email}.",
    "Account holder: {name}\nAccount number: {account}\nIBAN: {iban}",
    "Tax ID {ssn} does not match the name {name} on file.",
    "Delivery for {first} at {street} failed on {weekday}; reschedule at {email}.",
    "Login failed 5 times for {email} from {ip}.",
    "Thanks {first}! Your {product} subscription renews on {date} and will be charged to card {card}.",
    "{name} was born on {dob} in {city}.",
    "Please verify: name {name}, passport {passport}, phone {phone}.",
    "Our {dept} team ships {product} orders within two days; contact sales for bulk pricing.",
    "The {product} costs {amount} and order {order} is still pending.",
    "Reminder: the {dept} offsite is on {weekday}. Bring your laptop.",
    "Hello {first} {last}, we noticed a payment of {amount} from account {account} on {date}.",
    "Can you change the delivery address on order {order} to {street}, {city} {zip}? - {first}",
    "Name: {name}; Email: {email}; Phone: {phone}; DOB: {dob}",
    "{name2} referred {name} for the {job} role; both live in {city}.",
    "Lost card {card}. Please block it. My name is {name} and my phone is {phone}.",
    "Driver's licence number {dl}, date of birth {dob}, address {street}, {zip} {city}.",
    "Ticket from {email}: the {product} stopped working after the update.",
)

_SLOT = re.compile(r"\{([a-z0-9]+)\}")


def _value(fake: Faker, slot: str, rng: random.Random) -> str:
    if slot in ("name", "name2"):
        return fake.name()
    if slot == "first":
        return fake.first_name()
    if slot == "last":
        return fake.last_name()
    if slot == "email":
        return fake.email()
    if slot == "phone":
        return fake.phone_number()
    if slot == "street":
        return fake.street_address()
    if slot == "city":
        return fake.city()
    if slot == "zip":
        return fake.postcode()
    if slot == "dob":
        d = fake.date_of_birth(minimum_age=18, maximum_age=90)
        return d.strftime(rng.choice(["%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d %B %Y", "%Y-%m-%d"]))
    if slot == "date":
        d = fake.date_between(start_date="-3y", end_date="+1y")
        return d.strftime(rng.choice(["%d/%m/%Y", "%B %d, %Y", "%d %B %Y", "%Y-%m-%d", "%A %d %B"]))
    if slot == "ssn":
        return fake.ssn()
    if slot == "passport":
        try:
            return fake.passport_number()
        except AttributeError:
            return fake.bothify("?#######").upper()
    if slot == "dl":
        return fake.bothify(rng.choice(["?#######", "??######", "#########", "?###-####-####"])).upper()
    if slot == "card":
        return fake.credit_card_number()
    if slot == "iban":
        return fake.iban()
    if slot == "account":
        return fake.bban()
    if slot == "ip":
        return fake.ipv4_public()
    if slot == "age":
        return str(rng.randint(18, 90))
    if slot == "order":
        return fake.bothify(rng.choice(["#######", "ORD-#####", "A##-####"]))
    if slot == "product":
        return rng.choice(["standing desk", "headphones", "router", "espresso machine", "monitor", "backpack",
                           "keyboard", "air purifier", "water filter", "phone case"])
    if slot == "job":
        return fake.job()
    if slot == "amount":
        return f"${rng.randint(5, 4000)}.{rng.randint(0, 99):02d}"
    if slot == "weekday":
        return rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    if slot == "dept":
        return rng.choice(["finance", "marketing", "legal", "operations", "design", "security"])
    raise KeyError(slot)


def render(template: str, fake: Faker, rng: random.Random) -> tuple[str, list[list]]:
    out: list[str] = []
    spans: list[list] = []
    pos = 0
    last = 0
    for m in _SLOT.finditer(template):
        lit = template[last:m.start()]
        out.append(lit)
        pos += len(lit)
        slot = m.group(1)
        val = _value(fake, slot, rng).replace("\n", ", ")
        out.append(val)
        typ = SLOTS[slot]
        if typ is not None:
            spans.append([pos, pos + len(val), typ])
        pos += len(val)
        last = m.end()
    out.append(template[last:])
    return "".join(out), spans


def generate(n: int, seed: int = 20261016) -> list[dict]:
    rng = random.Random(seed)
    fakers = {}
    for loc, _ in LOCALES:
        f = Faker(loc)
        f.seed_instance(seed + len(fakers))
        fakers[loc] = f
    locs = [loc for loc, _ in LOCALES]
    weights = [w for _, w in LOCALES]
    docs = []
    for i in range(n):
        loc = rng.choices(locs, weights)[0]
        template = TEMPLATES[rng.randrange(len(TEMPLATES))]
        text, spans = render(template, fakers[loc], rng)
        docs.append({"id": f"faker:{i}", "source": "faker", "text": text, "spans": spans, "keep": [],
                     "meta": {"locale": loc}})
    return docs
