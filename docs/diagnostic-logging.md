# Registro de diagnóstico

En la WebUI, en **Ajustes → Configuración de interfaz → Registro de diagnóstico**, elige el nivel de registro, activa si lo necesitas
**Rastrear parámetros de las solicitudes a la API** y pulsa **Aplicar ajustes de registro**. Los ajustes se aplican de inmediato al servicio actual,
incluidos los logs posteriores de las tareas en ejecución; todas las sesiones del navegador del mismo servicio comparten esos ajustes.
Activar «Ocultar registro» solo oculta los logs de la interfaz y no desactiva la salida del terminal.

La API, la CLI y la WebUI usan la misma configuración:

```toml
[diagnostics]
log_level = "INFO"
log_requests = false
```

`log_level` admite TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR y CRITICAL.
Si se omite, se usa el `log_level` de nivel superior original, que por defecto es DEBUG. Tras editar el archivo a mano hay que reiniciar;
el botón «Aplicar» de la WebUI solo actualiza al momento el proceso actual y no modifica a distancia otros procesos de la API o la CLI.

El rastreo de solicitudes es un interruptor independiente y también se puede activar con el nivel ERROR. Está desactivado por defecto y registra:

- El ID de la solicitud, el método HTTP, la dirección censurada y los parámetros de la solicitud con longitud limitada.
- El estado devuelto (cuando se puede obtener), el tiempo empleado o el tipo de excepción del fallo.
- Los parámetros de las llamadas al SDK del LLM. Son la entrada del SDK, no el JSON en bruto que envía finalmente el SDK;
  los reintentos automáticos internos del SDK no se muestran uno a uno. En las solicitudes en streaming el tiempo solo se cuenta hasta obtener la respuesta o el objeto de flujo,
  sin incluir el tiempo de descarga posterior.

Cubre las solicitudes que el proyecto hace explícitamente con `requests` para material, vídeo, voz, publicación, inicio de sesión, etc., así como
las llamadas de texto compatibles con OpenAI, Azure OpenAI, Qwen, Gemini y LiteLLM.
No captura el propio tráfico de red del navegador, los cuerpos de la API entrantes, los subprocesos de la CLI, el WebSocket de Edge TTS
ni el transporte interno de otros SDK. Los logs normales de la aplicación y los niveles de log propios de los SDK de terceros tampoco son el mismo interruptor.

No se registran las cabeceras de las solicitudes, los cuerpos de las respuestas ni el contenido de los archivos subidos. Las claves, tokens, cookies, contraseñas y campos de firma
se ocultan, los valores de los parámetros de consulta de las URL se ocultan siempre, y el contenido de audio/vídeo/base64, etc., se omite. Los cuerpos de solicitud
en bruto no estructurados no se registran. **Los prompts y otros parámetros de negocio pueden aparecer en los logs**: al terminar de diagnosticar,
desactiva el rastreo, y antes de compartir logs comprueba igualmente si contienen contenido privado. El diagnóstico no envía logs al exterior
ni lanza, reintenta o repite solicitudes adicionales.

Los logs de la CLI se siguen enviando a stderr, y stdout mantiene un JSON legible por máquina. Los logs de tareas de la WebUI mantienen el filtrado actual por
pertenencia a hilo y el límite de líneas, y cambiar de nivel no elimina el manejador de logs de las tareas en ejecución.
