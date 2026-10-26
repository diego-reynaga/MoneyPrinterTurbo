# Servidor de imágenes local para `openai_image`

El material de stock a menudo tiene poco que ver con el guion. La fuente `openai_image` lo resuelve generando una imagen por cada fragmento del guion. Esta carpeta contiene un pequeño servidor que lo hace en tu propia GPU, así que no hay costes por imagen.

Expone la API `/v1/images/generations` de OpenAI y ejecuta [RealVisXL V5.0 Lightning](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning), un modelo SDXL fotorrealista que necesita unos 6 pasos por imagen.

Probado en una RTX 5070 Ti (16 GB):

- unos 3,9 s por imagen de 832x1216 por HTTP una vez en caliente (unos 2 s de eso son el propio modelo)
- de 5 a 7 minutos para un vídeo de 50 s con 8 escenas, incluidos la voz, los subtítulos de Whisper y el renderizado; la locución en línea explica la mayor parte de la variación

## Instalación

```bash
cd docs/local-image-server
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python server.py
```

`uvicorn server:app --port 7861` funciona igual: el worker de inactividad arranca con la aplicación en ambos casos.

La primera solicitud descarga los pesos fp16, unos 6,6 GB. El servidor escucha en `http://127.0.0.1:7861`. Un `size` que no sea `ANCHOxALTO` con números positivos recibe una respuesta 422.

Tests, desde la raíz del repositorio y sin necesidad de GPU: `python -m pytest test/services/test_local_image_policy.py test/services/test_local_image_server.py`. Los tests del servidor ejecutan la aplicación HTTP real con un pipeline sustituto y un Ollama falso.

## config.toml

```toml
[app]
video_source = "openai_image"
openai_image_base_url = "http://127.0.0.1:7861/v1"
openai_image_model = "realvisxl-v5-lightning"
match_materials_to_script = true
video_clip_duration = 6
```

`match_materials_to_script` pide al LLM hasta 8 palabras clave en el orden de la narración, lo que da una escena por cada parte del guion. Cada imagen se convierte en un clip, así que 8 palabras clave x 6 s cubren unos 48 s de narración. Con clips más cortos, las escenas se repiten para llenar el hueco.

## Compartir la GPU con Ollama

Medido en la RTX 5070 Ti: los pesos de SDXL ocupan 6,7 GB, y pintar una imagen de 832x1216 alcanza un pico de 10,6 GB reservados. qwen3:14b en Ollama ocupa 9,0 GB. No caben juntos en una tarjeta de 16 GB, así que el servidor mantiene solo uno de ellos en la GPU a la vez:

- Antes de cada imagen, pregunta a Ollama qué modelos están cargados, los descarga y espera hasta que Ollama informa de que ya no están. Esto se ejecuta en cada solicitud, no solo en la primera, porque MPT puede volver a cargar el LLM entre imágenes o entre vídeos consecutivos.
- Tras cada solicitud libera los búferes en caché de PyTorch. Sin eso, el proceso mantiene unos 14 GB reservados entre imágenes.
- Tras `IMAGE_IDLE_SECONDS` (90 por defecto) sin solicitudes, un worker en segundo plano mueve SDXL a la CPU, para que Ollama tenga toda la GPU para el siguiente guion.

El servidor no puede impedir que Ollama se cargue mientras se pinta una imagen. No se admite ejecutar el trabajo del LLM y el de las imágenes en el mismo momento: la siguiente solicitud de imagen recupera la GPU, pero mientras tanto ambos modelos la comparten. En Windows el controlador desborda el exceso a la memoria del sistema, lo que es lento pero no falló en las pruebas. Este caso no lo hemos probado en Linux.

## Opciones

| Variable | Valor por defecto | |
|---|---|---|
| `IMAGE_MODEL` | `SG161222/RealVisXL_V5.0_Lightning` | cualquier repositorio SDXL con pesos fp16 |
| `IMAGE_STEPS` | `6` | |
| `IMAGE_GUIDANCE` | `1.5` | |
| `IMAGE_IDLE_SECONDS` | `90` | tiempo de inactividad antes de mover SDXL a la CPU |
| `IMAGE_IDLE_CHECK_SECONDS` | `10` | cada cuánto comprueba el worker de inactividad |
| `IMAGE_PORT` | `7861` | lo usa `python server.py` |
| `OLLAMA_URL` | `http://localhost:11434` | |
| `OLLAMA_UNLOAD_TIMEOUT` | `30` | cuánto se espera a que Ollama libere la GPU |
| `CONTENT_POLICY` | `policy.toml` | ruta a un archivo de política, u `off` |

## Política de contenido

Las palabras clave de las escenas vienen de un LLM, así que un vídeo sobre religión o historia puede pedir al modelo de imágenes cosas que no deberían generarse. Antes de que cada prompt llegue al modelo, `policy.toml` lo comprueba. La política está activada por defecto, y una regla solo actúa cuando aparecen sus palabras clave.

| Regla | Qué hace |
|---|---|
| `god` | Dios nunca se muestra como una figura. La escena pasa a ser luz atravesando las nubes. |
| `prophet_muhammad`, `islamic_prophets`, `companions_and_family_of_the_prophet` | Sin representación, siguiendo la práctica islámica. La escena pasa a ser un paisaje vacío del lugar mencionado (La Meca, Medina, Hira, Sinaí, Jerusalén) o, si no, un desierto al amanecer. |
| `bahai_central_figures`, `sikh_gurus` | Sin representación. La escena pasa a ser un lugar adecuado sin personas. |
| `atrocities` | Sin "fotos históricas" inventadas de atrocidades reales. La escena pasa a ser un homenaje con velas. |
| `children_in_distress` | La escena pasa a ser una imagen simbólica sin personas. |
| `living_public_figures` | Sin caras fotorrealistas de cargos públicos con nombre. La escena pasa a ser un edificio gubernamental. |
| `sacred_script` | Los modelos de imágenes escriben árabe, hebreo, sánscrito o gurmukhi falsos, así que la caligrafía pasa a ser patrones geométricos. |
| `holy_books` | Los libros sagrados se muestran cerrados y sin letras. |
| `revered_figures` | Jesús, María, Buda, las deidades hindúes y otros sí se representan en estas tradiciones, así que se mantienen, con un encuadre respetuoso. |

`islamic_prophets_english_names` está desactivada por defecto. Para contenido islámico, actívala para que Jesús, Moisés, Abraham y Noé también se muestren como lugares en lugar de personas.

Sustituir la escena entera es deliberado. Un prompt negativo por sí solo no mantiene a las personas fuera de forma fiable: en las pruebas siguió produciendo figuras en 2 de 6 escenas de profetas. Subir la intensidad de la guía para compensarlo quemaba las imágenes del modelo Lightning.

Límites conocidos: las reglas por palabras clave no detectan las formulaciones que no incluyen y pueden dispararse por error con nombres comunes (Muhammad Ali, una persona llamada Isa). `test/services/test_local_image_policy.py` cubre los casos esperados. Edita `policy.toml` para adaptarla a tu público. Los patrones y las escenas de sustitución están en inglés porque las palabras clave de material que genera el LLM y los prompts que mejor entienden los modelos SDXL son en inglés.
