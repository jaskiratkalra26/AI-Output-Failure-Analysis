import unittest
import sqlite3
import os
import sys
import pandas as pd

# Add parent directory to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from claim_extraction.extractor import ClaimExtractor

class TestClaimExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test configuration and cleanup content."""
        print("\n[Setup] Initializing Test Environment...")
        cls.config_path = "tests/test_config.yaml"
        cls.db_path = "data/test_phase2_claims.db"
        
        # Ensure clean state
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)
            print(f"[Setup] Removed existing test database: {cls.db_path}")

        # Initialize Extractor with Test Config
        cls.extractor = ClaimExtractor(config_path=cls.config_path)

    def test_pipeline_execution(self):
        """1. Test that the full pipeline runs without errors."""
        print("\n[Test] Running Extraction Pipeline...")
        try:
            self.extractor.run()
        except Exception as e:
            self.fail(f"Pipeline execution failed with error: {e}")

    def test_database_creation(self):
        """2. Verify that the database file was created."""
        print("\n[Test] Verifying Database Creation...")
        self.assertTrue(os.path.exists(self.db_path), "Database file was not created.")

    def test_z_claims_stored(self):
        """3. Verify that claims were actually stored in the database."""
        print("\n[Test] Verifying Claim Storage...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check total count
        cursor.execute("SELECT COUNT(*) FROM claims")
        count = cursor.fetchone()[0]
        print(f"      Found {count} claims in database.")

        # Re-run if count is 0 because extraction runs in test_pipeline_execution
        # and test order is not guaranteed by default in unittest (though usually alphabetical)
        # However, run() accumulates results. 
        # If extraction failed (rate limits) we might have 0 claims.
        # But we saw logs: "Extraction complete. Total claims stored: 18"
        # Wait... the logs appear under "test_pipeline_execution".
        # Why does this test see 0?
        # A shared class instance is used, setupClass runs once.
        # But maybe the DB path is relative and CWD issues?
        # Or maybe connection needs to be closed?
        
        self.assertGreater(count, 0, "No claims were extracted to the database.")
        
        # Check structure
        cursor.execute("SELECT * FROM claims LIMIT 1")
        row = cursor.fetchone()
        self.assertIsNotNone(row, "Table is empty")
        # Schema: id, sample_id, claim_text, created_at
        self.assertEqual(len(row), 4, "Row does not have expected 4 columns")
        
        conn.close()

    def test_z_claims_content_validity(self):
        """4. Spot check extracted claims for validity (Atomic facts)."""
        print("\n[Test] Validating Claim Content...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fetch a few claims
        cursor.execute("SELECT sample_id, claim_text FROM claims LIMIT 5")
        rows = cursor.fetchall()
        
        for sample_id, claim in rows:
            print(f"      [Sample {sample_id}] Claim: {claim}")
            self.assertIsInstance(claim, str)
            self.assertGreater(len(claim), 5, "Claim text is suspiciously short")
            # Heuristic: Claims generally don't start with "I think" or "Maybe" if prompt worked
            self.assertFalse(claim.lower().startswith("i think"), "Claim appears to be an opinion")

        conn.close()

    def test_z_linkage_integrity(self):
        """5. Ensure all stored claims link back to valid Sample IDs from CSV."""
        print("\n[Test] Verifying Data Integrity...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unique sample_ids from DB
        cursor.execute("SELECT DISTINCT sample_id FROM claims")
        db_ids = {row[0] for row in cursor.fetchall()}
        
        # Get IDs from CSV
        df = pd.read_csv("phase1_llm_outputs.csv")
        csv_ids = set(df['id'].unique())
        
        # Check that DB IDs are a subset of CSV IDs
        invalid_ids = db_ids - csv_ids
        if invalid_ids:
            self.fail(f"Found claims linked to non-existent sample IDs: {invalid_ids}")
            
        print(f"      Verified {len(db_ids)} unique samples have claims.")
        conn.close()

if __name__ == '__main__':
    unittest.main()
