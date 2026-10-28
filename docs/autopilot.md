# Publicación automática diaria

Publica vídeos en tus redes conectadas todos los días sin que nadie intervenga: con el
nicho configurado una sola vez, el planificador genera y sube **3 vídeos al día a horas
aleatorias entre las 10:00 y las 22:00** (valores por defecto, todos configurables).

> **Está desactivada por defecto y no cambia nada del flujo actual.** Mientras
> `autopilot.enabled` sea `false`, generar vídeos desde la WebUI, la API o la CLI se
> comporta exactamente igual que antes. Puedes volver al trabajo manual en cualquier
> momento sin desinstalar ni deshacer nada.

## Cómo funciona

1. **Cada día se sortea un plan nuevo.** Al empezar el día, el planificador elige al
   azar tantas horas como vídeos hayas pedido dentro de la ventana configurada,
   respetando una separación mínima entre publicaciones. Dos días nunca publican al
   mismo ritmo.
2. **En cada hueco se elige un tema.** Con un nicho configurado, el LLM propone un
   ángulo nuevo evitando los temas ya publicados; con una lista de temas fija, se van
   rotando en orden.
3. **Se genera el vídeo con el flujo normal.** Es el mismo pipeline que usa la WebUI, con
   los mismos ajustes guardados de material, voz, subtítulos y música.
4. **Se publica y se espera la confirmación.** La subida va por Upload-Post a las
   plataformas que tengas conectadas, y el hueco solo se marca como publicado cuando
   Upload-Post confirma la subida.
5. **El plan y el resultado quedan guardados** en `storage/autopilot/state.json`, así que
   reiniciar el equipo o el proceso no vuelve a sortear el día en curso ni republica un
   hueco ya publicado.

## Configuración

Se puede hacer todo desde la WebUI, en **Ajustes → Configuración de publicación
automática**: debajo de las credenciales de Upload-Post está el bloque «Publicación
automática diaria», con el nicho, los vídeos al día y la franja horaria. El panel avisa
de lo que falta por configurar y muestra el plan de hoy, pero no publica nada por su
cuenta: el que publica es el proceso del planificador, que se arranca aparte.

Si prefieres editar el archivo, son dos cosas en `config.toml`.

### 1. Las cuentas de publicación, en `[app]`

```toml
[app]
upload_post_enabled = true
upload_post_api_key = "tu-clave-de-upload-post"
upload_post_username = "tu-usuario-de-upload-post"
upload_post_platforms = ["tiktok", "instagram", "youtube"]
```

Crea la cuenta y la clave en <https://upload-post.com/> y conecta allí tus redes.

**No hace falta activar `upload_post_auto_upload`.** La publicación automática encola su
propia subida, así que ese ajuste puede seguir en `false` para que los vídeos que generes
a mano desde la WebUI no se publiquen solos.

### 2. El nicho y el ritmo, en `[autopilot]`

```toml
[autopilot]
enabled = true
niche = "finanzas personales explicadas para gente que empieza a trabajar"
videos_per_day = 3
window_start_hour = 10
window_end_hour = 22
```

Si prefieres decidir tú los temas en lugar de dejárselo al LLM, usa una lista que se va
rotando día tras día:

```toml
subjects = [
  "Interés compuesto explicado con un ejemplo real",
  "Cómo montar un fondo de emergencia",
  "El presupuesto 50/30/20 en la práctica",
]
```

`config.example.toml` documenta toda la sección. Los ajustes que casi nunca hay que
tocar:

| Ajuste | Por defecto | Para qué sirve |
| --- | --- | --- |
| `min_gap_minutes` | `45` | Separación mínima entre dos publicaciones del día. Si no cabe en la ventana, se reduce automáticamente y se avisa en el log. |
| `reserve_minutes` | `0` | Minutos finales de la ventana en los que ya no se empieza nada. La hora sorteada es la de *inicio*; súbelo a `30` si quieres que también la subida termine dentro de la ventana. |
| `catch_up` | `true` | Publica un hueco cuya hora pasó mientras el equipo estaba apagado, siempre que siga dentro de la ventana. |
| `catch_up_grace_minutes` | `0` | Margen por detrás del fin de la ventana en el que todavía se acepta recuperar un hueco. Con `0`, nunca se publica fuera de la ventana. |
| `max_attempts` / `retry_delay_minutes` | `2` / `15` | Reintentos por hueco. Un reintento solo se programa si cabe dentro de la ventana del día. |
| `video_language` | `""` | Idioma del guion. Vacío deja que el LLM use el idioma del tema. |
| `cli_options` | `[]` | Opciones extra de `cli.py` aplicadas a cada hueco, por ejemplo `["--video-source", "pexels"]`. Sin nada aquí se usan los ajustes guardados por la WebUI. |
| `state_file` | `""` | Ruta del archivo de estado. Vacío usa `storage/autopilot/state.json`. |

Cada hueco publica **un** vídeo: `video_count` se fuerza a `1` aunque la WebUI tenga
guardado otro valor, para que el ritmo diario sea el que has pedido.

## Uso

Desde el código fuente:

```shell
uv run python autopilot.py check    # comprueba la configuración
uv run python autopilot.py plan     # muestra las horas sorteadas para hoy
uv run python autopilot.py test     # genera y publica un vídeo ya, para probar
uv run python autopilot.py run      # publica sin intervención, día tras día
uv run python autopilot.py status   # qué se ha publicado hoy y qué queda
uv run python autopilot.py --help
```

Con el paquete instalado con pip es lo mismo con `mpt autopilot <modo>`.

Todos los modos imprimen un objeto JSON en stdout y los logs van a stderr. El código de
salida es `0` si todo fue bien, `1` si hubo un problema de ejecución o faltan
credenciales, y `2` si los parámetros o la sección `[autopilot]` no son válidos. En el
modo `once` el código de salida describe solo esa pasada: un hueco que falló horas antes
no marca como fallidas las comprobaciones siguientes del día.

### Orden recomendado la primera vez

```shell
uv run python autopilot.py check   # ¿falta algo por configurar?
uv run python autopilot.py test    # ¿se genera y se publica de verdad?
uv run python autopilot.py plan    # ¿a qué horas publicará hoy?
uv run python autopilot.py run     # a partir de aquí ya no hace falta intervenir
```

El modo `test` publica de verdad un vídeo, fuera del plan del día, y no gasta ninguno de
los huecos. Es la forma de comprobar la cadena completa sin esperar a la primera hora
sorteada.

## Dejarlo publicando sin intervención

Hay dos formas, y las dos son válidas.

### Opción A: un proceso siempre en marcha (`run`)

El modo `run` se queda en un bucle, planifica cada día y publica a sus horas. Es el más
simple si el equipo o el servidor no se apaga.

En Docker, pega este servicio en el `docker-compose.yml` del proyecto, junto a `webui` y
`api`. Reutiliza el mismo montaje, así que comparte `config.toml` y `storage/` con ellos:

```yaml
  autopilot:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: "moneyprinterturbo-autopilot"
    command: [ "python3", "autopilot.py", "run" ]
    volumes: *common-volumes
    restart: always
```

No viene incluido de serie a propósito: así `docker compose up` no arranca un
planificador que nadie ha configurado todavía. Configúralo antes de añadirlo, porque con
`restart: always` y la configuración incompleta el contenedor entra en un ciclo de
reinicios registrando los problemas pendientes.

En Linux con systemd, un servicio `Restart=always` que ejecute
`uv run python autopilot.py run` con `WorkingDirectory` en el directorio del proyecto
hace lo mismo.

### Opción B: una tarea programada del sistema (`once`)

El modo `once` publica los huecos que ya han vencido y termina. Programar esa orden cada
pocos minutos es más robusto frente a reinicios y suspensiones del equipo, porque no
depende de que un proceso siga vivo.

En Windows, con el Programador de tareas, una tarea que se repita **cada 5 minutos**
ejecutando:

```
D:\ruta\al\proyecto\.venv\Scripts\python.exe D:\ruta\al\proyecto\autopilot.py once
```

con «Iniciar en» puesto en la carpeta del proyecto, y marcada para ejecutarse aunque el
usuario no haya iniciado sesión.

En Linux o macOS, lo equivalente en `crontab`:

```
*/5 * * * * cd /ruta/al/proyecto && .venv/bin/python autopilot.py once >> storage/autopilot/cron.log 2>&1
```

Comprobar cada 5 minutos no publica cada 5 minutos: solo se ejecutan los huecos cuya
hora ya ha llegado, y cada hueco se publica una sola vez. Si una generación dura más que
el intervalo, la comprobación siguiente ve que la anterior sigue en marcha, se salta la
pasada y sale con 0, de modo que la tarea programada no aparece como fallida mientras se
publica.

## Qué pasa cuando algo va mal

| Situación | Comportamiento |
| --- | --- |
| El equipo estaba apagado a la hora de un hueco | Con `catch_up = true` se publica en cuanto el planificador vuelve, si sigue dentro de la ventana; si no, el hueco se marca omitido y al día siguiente se planifica desde cero. |
| Falla la generación, o la plataforma rechaza el vídeo | Se reintenta tras `retry_delay_minutes`, hasta `max_attempts` veces, siempre que el reintento quepa en la ventana del día. Los demás huecos no se ven afectados. |
| La subida se queda sin confirmar (el proceso muere, o se agota `upload_timeout_seconds`) | Ese hueco se marca como fallido y **no** se reintenta: republicar un vídeo que quizá ya se subió es peor que perder un hueco. El `task_id` queda apuntado en el estado desde antes de empezar a generar, así que `status` te dice exactamente qué tarea revisar. |
| Se arranca el planificador a media tarde | El plan del día se sortea solo sobre las horas que quedan de ventana, en lugar de publicar de golpe las que ya pasaron. |
| Se arranca cuando la ventana ya cerró | Hoy no se publica nada y mañana se planifica el día completo. |
| Se arrancan dos planificadores a la vez | Un archivo de bloqueo junto al estado evita publicar el mismo hueco dos veces. En el modo `run` el segundo se niega a arrancar con error; en el modo `once` simplemente se salta la pasada y sale con 0, porque encontrarse la anterior todavía generando es lo normal. Si el anterior murió sin limpiar, el bloqueo se recupera solo. |
| El estado se corrompe en un corte de luz | Se vuelve a planificar el día. Se pierde el plan, no el día. |

`uv run python autopilot.py status` muestra en cualquier momento el plan de hoy, el
estado de cada hueco, el tema elegido, el `task_id` de la tarea y el error si lo hubo.
Los vídeos quedan en `storage/tasks/<task-id>/` como cualquier otra tarea, así que se
pueden revisar desde la gestión de tareas de la WebUI.

## Volver al trabajo manual

Pon `enabled = false` en `[autopilot]` y para el proceso o la tarea programada. Nada más:
la WebUI, la API y la CLI siguen funcionando igual, con los mismos ajustes guardados.
