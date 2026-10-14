"""Update a request so the patient can see progress on /my-requests.

    uv run python staff_update.py list
    uv run python staff_update.py update admin 3 in_progress --note "We are checking with the GP"
    uv run python staff_update.py update admin 3 done --note "Your fit note is ready to collect from reception"

Kinds: admin, prescription, callback, waiting. Statuses: in_progress, done ("done" also closes the request, so it
leaves the /staff page). The note is shown to the patient, so keep it short and kind. It only touches the demo database.

Why a command and not buttons on /staff: that page has no login, and a page that can change records must not
be open to everyone. A real product would put these actions in a proper staff tool behind a login, and would also
send the patient a text message or NHS App notification when the status changes.
"""

import argparse
import sys

import request_updates as ru
import surgery_clock as clock


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show what is still open, with the numbers to use")
    up = sub.add_parser("update", help="mark a request in progress or done")
    up.add_argument("kind", choices=list(ru.KINDS))
    up.add_argument("number", help="the request number from 'list'")
    up.add_argument("status", choices=list(ru.STATUSES))
    up.add_argument("--note", default="", help=f"a short message for the patient (up to {ru.NOTE_LIMIT} characters)")
    args = parser.parse_args(argv)

    if args.command == "list":
        items = ru.open_items()
        if not items:
            print("Nothing is open.")
            return 0
        print(f"{'KIND':13} {'NO.':>4}  {'PATIENT':18} WHAT")
        for item in items:
            print(f"{item['kind']:13} {item['id']:>4}  {item['patient'][:18]:18} {item['what'][:70]}")
        return 0
    try:
        result = ru.add_update(args.kind, args.number, args.status, args.note)
    except (ValueError, LookupError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"Updated {result['kind']} request {result['request_id']} to '{result['status']}'.")
    if result["note"]:
        print(f"The patient will see: {result['note']}")
    print("They will see it on /my-requests the next time they look.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
