import text2story as t2s
import os
import time
from pathlib import Path
import argparse
from datetime import datetime

EXPORT_DIR = os.path.join(Path(__file__).parent.parent, "Data", "auto_ann")
DATA_DIR = os.path.join(Path(__file__).parent.parent, "Data", "input_files")


if __name__ == '__main__':
    # Arguments for CMD
    parser = argparse.ArgumentParser(
        description="Extracts the narrative from a document (tweets) into an annotation file (.ann)"
    )

    parser.add_argument(
        "Filename", metavar="Filename",
        help="Filename of document that you want to extract the narrative from (must be in Data/input_files/)"
    )
    parser.add_argument("-o", "--outputname", nargs="?", metavar="string", default="your_output.ann", required=False,
                        help="Output name for the extracted narrative annotation file (exported to Data/auto_ann/)")

    args = parser.parse_args()

    start = time.time()
    t2s.start()

    with open(os.path.join(DATA_DIR, args.Filename), "r+", encoding="utf-8") as f:
        text = f.read()

    doc = t2s.Narrative("en", text, datetime.now().date().isoformat())

    doc.extract_actors()
    doc.extract_times()
    doc.extract_objectal_links()
    doc.extract_events()
    doc.extract_semantic_role_links()

    annotation = doc.ISO_annotation()
    with open(os.path.join(EXPORT_DIR, args.outputname), "w", encoding="utf-8") as f:
        f.write(annotation)
    f.close()

    print(f"Exported file - {args.outputname}")
    end = time.time()
    print(f"Computation time - {round(end - start, 2)} seconds")
