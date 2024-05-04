import { createClient } from "@supabase/supabase-js";

const supabaseUrl = "https://dislxzujovmrcvsalzdj.supabase.co";
const supabaseKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRpc2x4enVqb3ZtcmN2c2FsemRqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTQ4MTAwNTYsImV4cCI6MjAzMDM4NjA1Nn0.NxOI6ML5vvDF2pFhUYTfEIW-AbBgatSIwJdYYC6AVgU";

export const supabase = createClient(supabaseUrl, supabaseKey);