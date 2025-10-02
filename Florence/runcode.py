from pyngrok import ngrok
import uvicorn
import nest_asyncio
from google.colab import userdata

# Authenticate ngrok using Colab secrets
authtoken = userdata.get('NGROK_AUTHTOKEN')
ngrok.set_auth_token(authtoken)

# Connect and run
ngrok.kill()
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run('main:app', port=8000)