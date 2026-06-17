#!/bin/bash
# dbqueries.sh
# Creates the necessary schema, table, indexes, and constraints for the Prompt Insights Java API.
# Usage: ./dbqueries.sh
# Ensure you have set the necessary PG variables (e.g., PGUSER, PGPASSWORD, PGHOST, PGPORT) before running.
# The script connects specifically to the 'abc' database.

DB_NAME="abc"

echo "Connecting to database '$DB_NAME' to create schema and tables..."

psql -d "$DB_NAME" -c "
-- 1. Create the schema
CREATE SCHEMA IF NOT EXISTS ai;

-- 2. Create the table with BIGSERIAL for auto-incrementing ID
CREATE TABLE IF NOT EXISTS ai.prompts (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    category VARCHAR(40000) NOT NULL,
    sub_category VARCHAR(40000) NOT NULL,
    count BIGINT NOT NULL
);

-- 3. Create performance indexes for Spring Boot API queries
-- Index on date and category together
CREATE INDEX IF NOT EXISTS idx_prompts_date_category 
ON ai.prompts(date, category);

-- Index on date, category, and sub_category together
CREATE INDEX IF NOT EXISTS idx_prompts_date_cat_subcat 
ON ai.prompts(date, category, sub_category);

-- 4. Create Unique Constraint to prevent duplicate insertions
ALTER TABLE ai.prompts 
ADD CONSTRAINT unique_date_cat_subcat UNIQUE (date, category, sub_category);
"

if [ $? -eq 0 ]; then
    echo "Successfully created schema, table, indexes, and constraints in '$DB_NAME'."
else
    echo "Error executing database queries."
fi
