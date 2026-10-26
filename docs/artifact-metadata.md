# Consultar los metadatos de los artefactos

Los endpoints de artefactos aceptan `HEAD` además de `GET`:

```sh
curl --head http://localhost:8080/api/v1/stream/TASK_ID/final-1.mp4
curl --head http://localhost:8080/api/v1/download/TASK_ID/final-1.mp4
```

HEAD devuelve la longitud del contenido y el tipo de medio del artefacto completo sin enviar
sus bytes. Las descargas también devuelven el nombre de archivo adjunto existente y las cabeceras
de metadatos del archivo. Si hay una clave API configurada, usa la misma autenticación que con GET.
Los archivos que faltan y las rutas fuera del almacenamiento de tareas mantienen sus códigos
de error actuales. Range e If-Range no seleccionan representaciones parciales con
HEAD; envía GET con Range para obtener parte de un stream. Un artefacto vacío tiene
longitud cero. Una solicitud de metadatos correcta no valida la decodificación del vídeo ni
garantiza que una solicitud posterior vaya a ver el mismo archivo.
