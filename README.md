# 🤖 BimBam Buy - Agente Documental con IA (RAG)

> **Proyecto desarrollado para el Challenge de Alura**

## 📌 Descripción General
**BimBam Buy** es una tienda online enfocada en la experiencia de compra digital ágil y segura. Este proyecto resuelve la problemática de la pérdida de tiempo en la búsqueda de información interna (políticas de devolución, garantías y reembolsos) mediante un **Agente de Inteligencia Artificial** basado en la arquitectura **RAG (Retrieval-Augmented Generation)**.

Los colaboradores pueden realizar preguntas en lenguaje natural y recibir respuestas precisas e inmediatas basadas únicamente en la documentación oficial, incluyendo la fuente exacta y número de página.

---

## 🏗️ Arquitectura de la Solución
El agente opera bajo el siguiente flujo de procesamiento:

1. **Carga y Fragmentación:** Se lee el archivo PDF `Manual de Garantía de Productos de BimBam Buy.pdf` usando `PyPDFLoader` y se divide en fragmentos mediante `RecursiveCharacterTextSplitter`.
2. **Generación de Embeddings:** Se convierten los fragmentos de texto en vectores numéricos utilizando el modelo local `sentence-transformers/all-MiniLM-L6-v2`.
3. **Base de Datos Vectorial:** Se almacenan y buscan los fragmentos más relevantes usando `FAISS`.
4. **Razonamiento y Respuesta:** Se consultan los fragmentos recuperados en el LLM **Llama 3.3 70B** a través de la API de **Groq** para generar la respuesta final.
5. **Interfaz de Usuario:** Construida en **Streamlit** con un panel interactivo de chat, memoria de conversación y trazabilidad de fuentes.

---

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3.14
* **Framework IA:** LangChain
* **Embeddings:** HuggingFace / Sentence-Transformers
* **Base de Datos Vectorial:** FAISS
* **Modelo LLM:** Llama 3.3 70B (vía Groq API)
* **Interfaz Web:** Streamlit
* **Control de Versiones & Deploy:** GitHub & Streamlit Community Cloud

---

## 🚀 Instrucciones para Ejecución Local

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/joseordonez9/bimbam-buy-agente.git](https://github.com/joseordonez9/bimbam-buy-agente.git)
   cd bimbam-buy-agente

   ## ☁️ Evidencia del Deploy en la Nube

La aplicación se encuentra desplegada y ejecutable en la nube:

* 🔗 **Enlace público de la aplicación:** [https://bimbam-buy-agente-gxmyfgeghekow2s46jw8f2.streamlit.app/](https://bimbam-buy-agente-gxmyfgeghekow2s46jw8f2.streamlit.app/) *(reemplaza con tu URL real)*

### Captura de la aplicación en producción:
![Demostración del Agente](https://raw.githubusercontent.com/joseordonez9/bimbam-buy-agente/main/evidencia.png)