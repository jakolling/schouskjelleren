# Brewery Manager (Streamlit)

## Files
- `app.py` — the Streamlit app
- `requirements.txt` — Python dependencies
- `.streamlit/secrets.toml` — **NOT** committed to GitHub (configured in Streamlit Cloud)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud secrets (example)
In Streamlit Cloud → App → Settings → Secrets, paste something like:

```toml
DATABASE_URL = "postgresql+psycopg2://postgres:YOUR_PASSWORD@db.wvhflsjbbeqcxdmkncmz.supabase.co:5432/postgres"

[auth]
  [auth.cookie]
  name = "brewery_manager"
  key = "PUT_A_LONG_RANDOM_STRING_HERE"
  expiry_days = 30

  [auth.credentials]
    [auth.credentials.usernames]

      [auth.credentials.usernames.admin]
      name = "Admin"
      password = "PASTE_BCRYPT_HASH_HERE"
      role = "admin"

      [auth.credentials.usernames.viewer1]
      name = "Viewer 1"
      password = "PASTE_BCRYPT_HASH_HERE"
      role = "viewer"
```

### Generating password hashes
Use this one-time helper on your computer:

```python
import streamlit_authenticator as stauth
passwords = ["ADMIN_PASSWORD", "VIEWER_PASSWORD"]
print(stauth.Hasher(passwords).generate())
```

Copy the resulting hashes into the Secrets.
