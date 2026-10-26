# Proyectos de vídeo locales editables

Cambia una escena sin volver a renderizar el episodio entero. Esta CLI opcional
hace una instantánea del material local y la narración preparados, registra revisiones inmutables y
reutiliza los artefactos solo cuando coinciden tanto sus dependencias como su integridad SHA-256.
No hace llamadas a ninguna API ni publica vídeos. Los trabajos de generación existentes y el
comportamiento de la WebUI no cambian.

Es una base de revisiones/artefactos para las propuestas existentes de escenas/director,
no otro director ni un flujo con proveedores de pago. Primero genera o descarga el
material y la narración con el flujo de tareas existente o con tus herramientas preferidas.
Usa abajo las rutas locales de sus resultados. Mantén el ID de una escena al sustituir su material;
cambia el orden en el manifiesto para reordenar el vídeo.

## Requisitos

Python 3.11 o superior y FFmpeg/FFprobe en el PATH, incluido el codificador `libx264`.
Las entradas preparadas deben ser medios codificados autocontenidos: MOV/MP4, Matroska/WebM, AVI,
MPEG-TS/vídeo, Ogg, FLV, WAV, MP3, FLAC o AAC. Se rechazan las listas de reproducción HLS/DASH/concat y
los medios con referencias externas; copiar solo una lista de reproducción no haría una instantánea
de los segmentos a los que apunta. Concat solo se permite para las listas de montaje generadas
internamente. Los protocolos de red están desactivados para todas las entradas del renderizado.
No hay dependencias nuevas de Python. Los proyectos son almacenes locales de un solo host;
no se admiten sistemas de archivos de red con bloqueos poco fiables. Reserva disco suficiente
para las instantáneas de origen, las revisiones conservadas y las exportaciones. No hay poda automática
de artefactos ni borrado de exportaciones correctas antiguas.

## Crear, inspeccionar y renderizar

Guarda `episode.json` junto a tus recursos preparados:

```json
{
  "project_id": "product-episode",
  "settings": {
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "narration_volume": 1.0,
    "bgm_volume": 0.15
  },
  "scenes": [
    {
      "id": "opening",
      "footage": "opening.mp4",
      "audio": "opening.wav",
      "narration": "Presenta el producto."
    },
    {
      "id": "product-detail",
      "footage": "detail.mp4",
      "duration": 5
    }
  ],
  "bgm": "music.mp3"
}
```

```sh
python -m app.services.video_project --project storage/projects/product-episode revise episode.json
python -m app.services.video_project --project storage/projects/product-episode plan
python -m app.services.video_project --project storage/projects/product-episode render
python -m app.services.video_project --project storage/projects/product-episode status
```

Las rutas relativas de los recursos se resuelven respecto al directorio del manifiesto. Los originales se copian
dentro del proyecto; editar después el archivo original no cambia en silencio una
revisión. El ID de revisión que imprime `revise` identifica una instantánea de entrada
inmutable. `status` informa de las etapas completadas, los artefactos reutilizados, los fallos y la
ruta de exportación. `project.json` apunta a la revisión actual y a la última exportación
correcta, que se conserva hasta que una revisión actual más nueva tenga éxito.

El audio determina la duración de las escenas narradas. Las escenas sin audio usan `duration`
(cinco segundos por defecto). El material corto se repite en bucle; el largo se recorta. El audio del
vídeo de origen se sustituye intencionadamente por la narración preparada o por silencio.
Todas las salidas de escena usan las mismas dimensiones, tasa de fotogramas y audio estéreo a 48 kHz.
La mezcla final de BGM repite la música en bucle hasta la duración de la línea de tiempo y no normaliza
la narración hasta hacerla desaparecer. Un volumen de BGM cero produce una exportación sin mezclar.

`narration` es un texto de procedencia, no una solicitud de TTS. Al cambiar la narración, aporta
también su nuevo audio preparado; la CLI no puede verificar que la grabación diga el
texto. Los subtítulos alineados por palabra, las transiciones, las superposiciones y la generación con proveedores
no están implementados en este primer renderizador local. La etapa de tiempos registra la duración,
no alineaciones de palabras generadas.

## Sustituir, comparar y reutilizar

Edita solo los campos de la escena/BGM afectados y vuelve a ejecutar `revise` con el mismo
`project_id` y los mismos ID de escena estables. Usa los ID antiguo y nuevo que se imprimen para comparar:

```sh
python -m app.services.video_project --project storage/projects/product-episode compare REVISION_ANTIGUA REVISION_NUEVA
python -m app.services.video_project --project storage/projects/product-episode plan
python -m app.services.video_project --project storage/projects/product-episode render
```

| Cambio | Etapas reconstruidas |
|---|---|
| El material de una escena | El renderizado de esa escena, el montaje y la exportación final |
| La narración/audio de una escena | La normalización de audio, los tiempos y el renderizado de esa escena; el montaje y la exportación |
| Archivo o volumen de la BGM | Solo la mezcla/exportación final |
| Orden de las escenas | Solo el montaje y la exportación |
| Dimensiones/tasa de fotogramas | Los renderizados de escena, el montaje y la exportación |
| Volumen de la narración | Audio, tiempos, escenas, montaje y exportación |
| Versión de FFmpeg/contrato del renderizador | Una revisión nueva se reconstruye con la huella de transformación cambiada |

El plan explica las dependencias cambiadas, los artefactos completados que faltan y los fallos
de integridad. Los artefactos dañados se reconstruyen a partir de instantáneas verificadas. Las instantáneas de origen
dañadas fallan de forma segura: crea una revisión nueva a partir del material preparado original.
Las cachés se quedan dentro de un proyecto; otro proyecto no puede reutilizarlas implícitamente.
No hay porcentajes de ahorro ni garantías de idempotencia de generación de pago.

## Cancelación, fallos y revisiones concurrentes

```sh
python -m app.services.video_project --project storage/projects/product-episode cancel
python -m app.services.video_project --project storage/projects/product-episode render --retry
```

Una solicitud de cancelación detiene el subproceso multimedia activo e impide la promoción.
Tras un renderizado fallido, cancelado o interrumpido hace falta un `--retry` explícito.
Las etapas completadas y verificadas siguen siendo reutilizables. Un fallo del proceso puede dejar el estado
`running`; el reintento vuelve a comprobar los artefactos en lugar de suponer que terminó. Los directorios
`.work` temporales sin confirmar que deja una muerte brusca del proceso pueden borrarse cuando
no hay ningún renderizado activo. Nunca son una fuente de artefactos reutilizables de confianza.

Solo un renderizador escribe una revisión a la vez. Revisiones distintas pueden renderizarse
a la vez y compartir artefactos completados inmutables. Si se crea una revisión más nueva
mientras se renderiza una antigua, la antigua puede terminar para poder compararla,
pero no puede sustituir la última exportación correcta del proyecto actual. Los bloqueos del sistema operativo se liberan
al morir el proceso. Cada comando multimedia externo tiene un plazo acotado (600
segundos por defecto, ajustable con `render --stage-timeout`).

Los manifiestos y los registros de artefactos usan la sustitución atómica de archivos. Los bytes de los artefactos se
publican antes que el registro de confirmación; nunca se confía en escrituras incompletas. Una
huella incluye el contrato del renderizador, las versiones de FFmpeg/FFprobe, los ajustes
relevantes y los hashes del contenido. Se rechazan los campos desconocidos del manifiesto/ajustes,
incluidas las credenciales de proveedores. Guarda aquí solo contenido de narración público; ningún
software puede deducir si un texto arbitrario escrito por el usuario contiene un secreto.

## Ajustes de renderizado reutilizables por episodio/marca

```sh
python -m app.services.video_project --project storage/projects/product-episode save-preset brand-render.json
python -m app.services.video_project --project storage/projects/next-episode revise next-episode.json --preset brand-render.json
```

Los preajustes reutilizan las dimensiones, la tasa de fotogramas y el volumen de la narración/BGM. No contienen
recursos, ID de proyecto, credenciales, elecciones de proveedores ni afirmaciones de personajes
generados con IA coherentes. Un preajuste proporcionado sustituye los ajustes del manifiesto.
Los recursos de las escenas y la narración siguen siendo explícitos en cada episodio.

## Límite de integración

`VideoProject.create_revision`, `plan`, `render`, `status` y `compare` ofrecen el
mismo contrato de medios preparados que la CLI. Un futuro director aceptado puede pasar sus
rutas de narración/material completadas, conservar los ID de escena y consumir los recibos de exportación/revisión
sin cambiar la propiedad de los proveedores. Los registros existentes de resultados creativos
pueden hacer referencia a un ID de revisión completada sin introducir publicación automatizada
ni tratar el rendimiento observado como prueba causal.
