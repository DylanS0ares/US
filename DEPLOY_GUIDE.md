# Deploy no Streamlit Cloud

## Passos:

1. **Crie um repositório no GitHub** e envie seu código:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/SEU_USUARIO/nome-do-repo.git
   git push -u origin main
   ```

2. **Acesse [share.streamlit.io](https://share.streamlit.io)**
   - Faça login com GitHub
   - Clique em "New app"
   - Selecione seu repositório
   - Deploy!

## Arquivos necessários já configurados:
- `requirements.txt` ✓
- `app.py` (ponto de entrada) ✓

## Vantagens:
- **Gratuito**
- **Sempre online**
- URL pública fixa
- Atualiza automático quando você faz push

---

# Deploy no Railway (Alternativa)

1. Crie conta em [railway.app](https://railway.app)
2. Conecte seu repositório GitHub
3. Railway detectará automaticamente o Dockerfile
4. Deploy automático!

---

# Deploy no Render (Alternativa)

1. Crie conta em [render.com](https://render.com)
2. New → Web Service → Connect GitHub repo
3. Configure:
   - **Runtime**: Docker
   - **Plan**: Free
4. Deploy!
