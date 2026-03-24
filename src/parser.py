import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import TWITTER_TIME_FMT

def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def extract_tweet_record(
    tweet: Dict[str, Any],
    event: str,
    label: str,
    thread_id: str,
    is_source: bool,
) -> Dict[str, Any]:
    created_at = datetime.strptime(tweet.get("created_at"), TWITTER_TIME_FMT)

    return {
        "event": event,
        "label": label,
        "thread_id": thread_id,
        "is_source": is_source,
        "tweet_id": str(tweet.get("id")),
        "user_id": str(tweet["user"]["id"]),
        "screen_name": tweet["user"]["screen_name"],
        "created_at": created_at,
        "created_at_ts": created_at.timestamp() if created_at else None,
        "text": tweet.get("text"),
        "in_reply_to_status_id": str(tweet.get("in_reply_to_status_id")),
        "in_reply_to_user_id": str(tweet.get("in_reply_to_user_id")),
        "retweet_count": tweet.get("retweet_count"),
        "favorite_count": tweet.get("favorite_count"),
        "lang": tweet.get("lang"),
    }


def parse_thread(thread_dir: Path, event: str, label: str) -> Dict[str, Any]:
    thread_id = thread_dir.name

    source_dir = p = thread_dir / "source-tweet"
    source_json = sorted(source_dir.glob("*.json"))[0]
    source_tweet = load_json_file(source_json)
    rows: List[Dict[str, Any]] = [
        extract_tweet_record(source_tweet, event, label, thread_id, is_source=True)
    ]

    reactions_dir = thread_dir / "reactions"
    if reactions_dir is not None:
        for reaction_file in sorted(reactions_dir.glob("*.json")):
            reaction_tweet = load_json_file(reaction_file)
            rows.append(
                extract_tweet_record(reaction_tweet, event, label, thread_id, is_source=False)
            )

    return {
        "thread_id": thread_id,
        "event": event,
        "label": label,
        "tweets": rows,
        "path": str(thread_dir),
    }


def parse_pheme_dataset(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    all_rows: List[Dict[str, Any]] = []

    for event_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        event = event_dir.name

        label_dirs = [p for p in event_dir.iterdir() if p.is_dir()]

        for label_dir in sorted(label_dirs):
            label = label_dir.name

            for thread_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                thread = parse_thread(thread_dir, event, label)
                all_rows.extend(thread["tweets"])

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df = df.sort_values(
            ["event", "label", "thread_id", "created_at_ts"],
            na_position="last"
        ).reset_index(drop=True)

    return df