import pandas as pd
from typing import Optional, Tuple

# Epoch definitions
UNIX_EPOCH = pd.to_datetime('1970-01-01 00:00:00')  # Unix epoch start date
GPS_EPOCH = pd.to_datetime('1980-01-06 00:00:00')   # GPS epoch start date
NI_EPOCH = pd.to_datetime('1904-01-01 00:00:00')  # NI epoch start date
# See https://www.ni.com/en/support/documentation/supplemental/08/labview-timestamp-overview.html

def parse_timestamp_with_epoch(
    timestamp: float,
    expected_year: int
) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
    """
    Convert a timestamp to datetime using different epoch assumptions.

    Args:
        timestamp: Timestamp value as a float (seconds since epoch)
        expected_year: The year that the parsed datetime should have

    Returns:
        Tuple of (epoch_name, parsed_datetime) if a matching epoch is found,
        otherwise (None, None)
    """
    # Define epochs as pandas timestamps
    epochs = {
        'unix': pd.Timestamp('1970-01-01 00:00:00', tz='UTC'),  # Unix epoch
        'gps': pd.Timestamp('1980-01-06 00:00:00', tz='UTC'),   # GPS epoch
        'mac': pd.Timestamp('1904-01-01 00:00:00', tz='UTC')    # Mac/HFS+ epoch
    }

    # Convert timestamp to timedelta
    td = pd.Timedelta(seconds=timestamp)

    print(f"\nParsing timestamp {timestamp} (expecting year {expected_year}):")
    print("-" * 60)

    matched_epoch = None
    matched_datetime = None

    # Test all epochs
    for epoch_name, epoch_start in epochs.items():
        try:
            # Add timedelta to epoch
            dt = epoch_start + td

            print(f"{epoch_name.upper():8} epoch: {dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")

            # Check if this yields the expected year
            if dt.year == expected_year:
                print(f"   Matches expected year {expected_year}")
                matched_epoch = epoch_name
                matched_datetime = dt

        except (ValueError, OverflowError) as e:
            print(f"{epoch_name.upper():8} epoch: Error - {str(e)}")

    if matched_epoch:
        print(f"\nReturning: {matched_epoch} epoch")
    else:
        print(f"\nNo epoch yielded year {expected_year}")

    return matched_epoch, matched_datetime