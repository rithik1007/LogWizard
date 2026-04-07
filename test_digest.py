"""Quick test: send a single digest email."""
import sys
sys.path.insert(0, ".")

from datetime import datetime
from echelon.config import settings
from echelon.sources.file_source import FileSource
from echelon.digest import subscribe, summarise_day, send_digest_email

email = "rithik.dhariwal@wolterskluwer.com"
subscribe(email)
print(f"Subscribed: {email}")

source = FileSource()
# Use March 31 (the date in structured_events.json) so we get actual log data
summary = summarise_day(source, target_date=datetime(2026, 3, 31))
print(f"Date: {summary['date']}")
print(f"Total logs: {summary['total']}")
print(f"Errors: {len(summary['errors'])}")
print(f"Warnings: {len(summary['warnings'])}")
print(f"Apps: {list(summary['apps'].keys())}")
for app, data in summary['apps'].items():
    print(f"  {app}: {data['status']} ({data['total']} logs, {data['error_count']} errors)")

print("\nSending digest email...")
failed = send_digest_email([email], summary)
if not failed:
    print("SUCCESS — Check your Outlook inbox!")
else:
    print(f"FAILED for: {failed}")
