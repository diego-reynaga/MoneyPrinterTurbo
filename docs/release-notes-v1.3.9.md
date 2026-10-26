<h2 id="es-v1.3.9">Novedades destacadas</h2>

- Se añade el **paquete oficial `mpt` para pip**. Arranca la WebUI con `mpt webui` o genera vídeos con `mpt cli`.
- Se añaden **efectos de imagen estática y de zoom lento** para imágenes locales y generadas con IA. El zoom lento sigue siendo el valor por defecto.
- Se añade el acceso con la suscripción de Kimi Code mediante autorización en el navegador, junto con mejoras en el diálogo de inicio de sesión, las actualizaciones de estado y la cancelación.
- Se añade un **servidor de imágenes SDXL local** opcional para el flujo de material de imagen compatible con OpenAI, que libera la GPU cuando está inactivo.
- Se añaden los proveedores LLM Opper, onomeo, Muse Spark y EmpirioLabs AI, y se completan las traducciones de la WebUI al portugués de Brasil.
- Se añaden **controles de registro de diagnóstico** en la WebUI, incluidos los cambios del nivel de registro en tiempo de ejecución y el rastreo opcional de solicitudes a la API con censura y límites de tamaño.

## Correcciones y mejoras

- Se corrigen el brillo de las imágenes en escala de grises, los fondos transparentes de las imágenes, la alineación de la máscara del zoom y los colores de fondo de los subtítulos. Se mejoran las transiciones en clips cortos, el fundido de salida de la música de fondo y el tratamiento de caracteres especiales en las rutas de montaje.
- Se aíslan los archivos de imagen intermedios entre trabajos concurrentes, se mejora la identidad de la caché de imágenes y se validan las descargas en caché. Los vídeos completados se conservan cuando falla otro vídeo del mismo lote.
- Se corrigen la velocidad, el volumen y el tratamiento del contenedor de audio en varios proveedores de voz. Se mejora la validación del audio remoto y se conserva el audio existente hasta que la salida que lo sustituye está lista.
- Se mejoran el redondeo de las marcas de tiempo de los subtítulos, el ajuste de palabras largas, el tratamiento de Unicode y el reconocimiento de las etiquetas de pausa. Las líneas formadas solo por símbolos y los separadores ya no distorsionan la alineación estimada de los subtítulos.
- Se corrige que los títulos sociales de respaldo en chino y japonés usaran el guion entero. Se mejoran la extracción de JSON, la eliminación del razonamiento, la limpieza de Markdown y la validación de los términos de búsqueda generados.
- Se amplían los formatos de subida local admitidos, incluidos los JPEG de cámara con varias imágenes. Se mejoran en la WebUI las comprobaciones del historial de tareas, el estado de borrado, las vistas previas de subtítulos y los mensajes de error de conexión con el modelo.
- Se añade soporte de HEAD para los artefactos generados y se mejoran las descargas por rangos de bytes, los errores de validación, y las actualizaciones y protecciones de borrado de tareas de Redis.
- Se reducen los reenvíos de voz inseguros tras respuestas perdidas o redirecciones, se conservan los identificadores de seguimiento de Upload-Post y se mejora la detección de errores de envío de WaveSpeed.
- Se actualiza Edge TTS para resolver los fallos 403, se restringe PyAV por compatibilidad con Whisper y se corrigen la limpieza de caché en Windows y el tratamiento del BOM UTF-8 en los archivos de idioma.
- Se mejoran el tratamiento de la configuración y las comprobaciones previas del helper del agente, incluidos los flujos con narración e imágenes ya preparadas que no necesitan un LLM.

## Notas de uso y actualización

- El paquete oficial de PyPI es **`mpt`**, mantenido por este proyecto, y requiere **Python 3.11 o superior**. Instálalo o actualízalo con `python -m pip install --upgrade mpt`; se recomienda un entorno virtual. El paquete `MoneyPrinterTurbo`, con otro nombre, es una distribución de terceros.
- La instalación con pip guarda por defecto la configuración y los archivos generados en tu directorio de usuario; define `MPT_HOME` para elegir otra ubicación. Las instalaciones desde el código fuente siguen usando el directorio del proyecto. Al cambiar de método de instalación, migra tu configuración y los recursos necesarios al directorio de ejecución elegido.
- El movimiento de imágenes usa por defecto el **zoom lento** para los usuarios existentes. Para una imagen fija sin efectos de transición, elige **Estática** en el movimiento de imágenes y **Ninguna** en las transiciones. El ajuste está disponible en la WebUI, la CLI y la API.
- Kimi Code requiere iniciar sesión en el navegador y usa los límites de uso de tu plan. Los demás proveedores LLM nuevos necesitan sus credenciales de API y su configuración correspondientes.
- El servidor SDXL local necesita dependencias propias, descargas de modelos y un servicio de GPU en marcha; no arranca automáticamente con MPT. Su configuración por defecto descarga los modelos de Ollama antes de generar imágenes, así que evita ejecutar a la vez trabajo del LLM y de imágenes en la misma GPU. Consulta las [instrucciones de instalación](https://github.com/harry0703/MoneyPrinterTurbo/blob/v1.3.9/docs/local-image-server/README.md).
- El rastreo de solicitudes a la API está **desactivado por defecto**. Con él activado pueden aparecer prompts y parámetros de negocio en los logs; desactívalo al terminar de diagnosticar y comprueba que no haya contenido privado antes de compartir logs. Consulta la [guía de registro de diagnóstico](https://github.com/harry0703/MoneyPrinterTurbo/blob/v1.3.9/docs/diagnostic-logging.md).

## Cambios

- Paquete oficial para pip con lanzadores de WebUI y CLI — [c3634de](https://github.com/harry0703/MoneyPrinterTurbo/commit/c3634de097ea4951935d22e94580ff0ad2665503).
- Opciones de imagen estática y zoom lento — [90f7701](https://github.com/harry0703/MoneyPrinterTurbo/commit/90f77011d7e3b46d382e6d0180dabfb6725ffee4).
- Proveedor de la suscripción de Kimi Code — @ysntony en [#1599](https://github.com/harry0703/MoneyPrinterTurbo/pull/1599), seguido de [mejoras en el flujo de inicio de sesión](https://github.com/harry0703/MoneyPrinterTurbo/commit/4204d0f).
- Servidor de imágenes SDXL local — @imkashif en [#1668](https://github.com/harry0703/MoneyPrinterTurbo/pull/1668).
- Proveedor LLM Opper — @Felixkw12 en [#1589](https://github.com/harry0703/MoneyPrinterTurbo/pull/1589).
- Proveedor LLM onomeo — @Owaid0723666 en [#1600](https://github.com/harry0703/MoneyPrinterTurbo/pull/1600).
- Proveedor Muse Spark y traducciones al portugués de Brasil — @humbertocortezia en [#1657](https://github.com/harry0703/MoneyPrinterTurbo/pull/1657).
- Proveedor EmpirioLabs AI — @Adam-Dalloul en [#1694](https://github.com/harry0703/MoneyPrinterTurbo/pull/1694).
- Registro de diagnóstico en tiempo de ejecución y rastreo de solicitudes — [6940611](https://github.com/harry0703/MoneyPrinterTurbo/commit/6940611cc2791d1ca13b3af7ee893ccd9544b874).
- Títulos sociales de respaldo en chino y japonés — [e25326e](https://github.com/harry0703/MoneyPrinterTurbo/commit/e25326e).
- Conservación de los vídeos completados tras fallos en un lote — @hdu-r3drum en [#1597](https://github.com/harry0703/MoneyPrinterTurbo/pull/1597).
- Corrección del 403 de Edge TTS — @HEOJUNFO en [#1652](https://github.com/harry0703/MoneyPrinterTurbo/pull/1652).
- Compatibilidad de Whisper/PyAV — @cristianchiriac en [#1582](https://github.com/harry0703/MoneyPrinterTurbo/pull/1582).
- Limpieza de caché en Windows — @shan3520 en [#1654](https://github.com/harry0703/MoneyPrinterTurbo/pull/1654).
- Soporte de BOM UTF-8 en los archivos de idioma — @ketpatil77 en [#1658](https://github.com/harry0703/MoneyPrinterTurbo/pull/1658).
- Protección de tareas con estados que no se pueden analizar — @RerankerGuo en [#1663](https://github.com/harry0703/MoneyPrinterTurbo/pull/1663).
- Errores de conexión con el modelo más claros — @fengqiaoqu en [#1645](https://github.com/harry0703/MoneyPrinterTurbo/pull/1645).
- Comprobaciones previas del material local — @drakeo338 en [#1585](https://github.com/harry0703/MoneyPrinterTurbo/pull/1585); conservación de los ID de tareas de Redis — @ac1982 en [#1588](https://github.com/harry0703/MoneyPrinterTurbo/pull/1588).
- Renderizado en escala de grises, solicitudes HEAD de artefactos y flujos del agente con medios preparados — @rudycelekli en [#1620](https://github.com/harry0703/MoneyPrinterTurbo/pull/1620), [#1636](https://github.com/harry0703/MoneyPrinterTurbo/pull/1636) y [#1608](https://github.com/harry0703/MoneyPrinterTurbo/pull/1608), junto con muchas correcciones de fiabilidad de subtítulos, voz, material, API y tareas.

¡Gracias a todas las personas que han contribuido y a todas las que han informado de problemas y ayudado a probar estos cambios!

**Registro de cambios completo**: [v1.3.8...v1.3.9](https://github.com/harry0703/MoneyPrinterTurbo/compare/v1.3.8...v1.3.9)
