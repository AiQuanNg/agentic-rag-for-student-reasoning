#!/usr/bin/env python3
"""
Generate embeddings for criteria and rubrics tables using all-MiniLM-L6-v2
"""
import os
import sys
from typing import List
import psycopg2
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'agentic_rag'),
    'user': os.getenv('DB_USER', 'rag'),
    'password': os.getenv('DB_PASSWORD', 'studentreasoning'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

# Initialize model
print("Loading all-MiniLM-L6-v2 model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")


def generate_criteria_embeddings(conn):
    """Generate embeddings for criteria table"""
    print("\n=== Processing Criteria Table ===")

    with conn.cursor() as cur:
        # Fetch all criteria
        cur.execute("""
                    SELECT id, name, description, guidance
                    FROM criteria
                    """)
        rows = cur.fetchall()

        if not rows:
            print("No criteria found in database.")
            return

        print(f"Found {len(rows)} criteria records")

        # Prepare texts for embedding
        texts = []
        ids = []
        for row in rows:
            id_, name, description, guidance = row
            # Concatenate fields with separators
            text_parts = [name or ""]
            if description:
                text_parts.append(description)
            if guidance:
                text_parts.append(guidance)

            text = ". ".join(text_parts)
            texts.append(text)
            ids.append(id_)

        # Generate embeddings in batch
        print("Generating embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True)

        # Update database
        print("Updating database...")
        update_data = [
            (embedding.tolist(), id_)
            for embedding, id_ in zip(embeddings, ids)
        ]

        execute_batch(
            cur,
            "UPDATE criteria SET embedding = %s::vector WHERE id = %s",
            update_data,
            page_size=100
        )

        conn.commit()
        print(f"✓ Updated {len(ids)} criteria records")


def generate_rubrics_embeddings(conn):
    """Generate embeddings for rubrics table"""
    print("\n=== Processing Rubrics Table ===")

    with conn.cursor() as cur:
        # Fetch all rubrics
        cur.execute("""
                    SELECT id, exemplar, descriptor
                    FROM rubrics
                    """)
        rows = cur.fetchall()

        if not rows:
            print("No rubrics found in database.")
            return

        print(f"Found {len(rows)} rubric records")

        # Prepare texts for embedding
        texts = []
        ids = []
        for row in rows:
            id_, exemplar, descriptor = row
            # Concatenate exemplar + descriptor (if present)
            text = exemplar or ""
            if descriptor:
                text = f"{descriptor}: {text}"

            texts.append(text)
            ids.append(id_)

        # Generate embeddings in batch
        print("Generating embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True)

        # Update database
        print("Updating database...")
        update_data = [
            (embedding.tolist(), id_)
            for embedding, id_ in zip(embeddings, ids)
        ]

        execute_batch(
            cur,
            "UPDATE rubrics SET embedding = %s::vector WHERE id = %s",
            update_data,
            page_size=100
        )

        conn.commit()
        print(f"✓ Updated {len(ids)} rubric records")


def main():
    """Main execution"""
    try:
        # Connect to database
        print("Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        print("✓ Connected")

        # Generate embeddings for both tables
        generate_criteria_embeddings(conn)
        generate_rubrics_embeddings(conn)

        print("\n=== All Done! ===")
        print("Embeddings generated and stored successfully.")

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    main()