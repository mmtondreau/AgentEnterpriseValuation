-- Initialize database for ADK agent state management
-- NOTE: Google ADK will automatically create its own tables (sessions, events, etc.)
-- This file only creates additional custom tables needed by the application

-- Create extension for UUID support
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


CREATE TABLE memory_entries (
  id SERIAL PRIMARY KEY,
  app_name TEXT NOT NULL,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  content TEXT NOT NULL,
  author TEXT,
  timestamp TIMESTAMPTZ
);

CREATE INDEX idx_memory_app_user
  ON memory_entries(app_name, user_id);

  CREATE TABLE api_cache (
  cache_key   text PRIMARY KEY,
  payload     jsonb NOT NULL,
  fetched_at  timestamptz NOT NULL,
  expires_at  timestamptz
);