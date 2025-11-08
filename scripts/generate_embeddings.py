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
            # Concatenate all text fields for comprehensive embedding
            parts = [name or ""]
            if description:
                parts.append(description)
            if guidance:
                parts.append(guidance)

            text = " | ".join(parts)  # Uses pipe separator for clarity
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


def generate_questions_embeddings(conn):
    """Generate embeddings for questions table"""
    print("\n=== Processing Questions Table ===")
    with conn.cursor() as cur:
        # Fetch all questions - ONLY use 'text' field
        cur.execute("""
                    SELECT id, text
                    FROM questions
                    WHERE text IS NOT NULL
                      AND trim(text) != ''
                    """)
        rows = cur.fetchall()

        if not rows:
            print("No questions found in database.")
            return

        print(f"Found {len(rows)} question records")

        # Prepare texts for embedding - use text field directly
        texts = []
        ids = []

        for row in rows:
            id_, text = row
            texts.append(text)  # Use question text as-is
            ids.append(id_)

        # Generate embeddings
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
            "UPDATE questions SET embedding = %s::vector WHERE id = %s",
            update_data,
            page_size=100
        )

        conn.commit()
        print(f"✓ Updated {len(ids)} question records")


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
            # Prioritize exemplar (student response example) for Extractor Agent
            parts = [exemplar or ""]
            if descriptor:
                parts.append(descriptor)  # exemplar FIRST, then descriptor

            text = " | ".join(parts)

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


def generate_keywords_embeddings(conn):
    """Generate embeddings for topic_keywords table"""
    print("\n=== Processing Topic Keywords Table ===")
    with conn.cursor() as cur:
        # Fetch approved keywords - ONLY use 'keyword' field
        cur.execute("""
                    SELECT id, keyword
                    FROM topic_keywords
                    WHERE status = 'approved'
                      AND keyword IS NOT NULL
                      AND trim(keyword) != ''
                    """)
        rows = cur.fetchall()

        if not rows:
            print("No approved keywords found in database.")
            return

        print(f"Found {len(rows)} keyword records")

        # Prepare texts for embedding - use keyword field directly
        texts = []
        ids = []

        for row in rows:
            id_, keyword = row
            texts.append(keyword)  # Use keyword as-is
            ids.append(id_)

        # Generate embeddings
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
            "UPDATE topic_keywords SET embedding = %s::vector WHERE id = %s",
            update_data,
            page_size=100
        )

        conn.commit()
        print(f"✓ Updated {len(ids)} keyword records")


def main():
    """Main execution"""
    try:
        # Connect to database
        print("Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        print("✓ Connected")

        # Generate embeddings for all tables
        generate_questions_embeddings(conn)
        generate_criteria_embeddings(conn)
        generate_rubrics_embeddings(conn)
        generate_keywords_embeddings(conn)

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