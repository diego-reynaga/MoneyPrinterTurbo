# Usa un runtime oficial de Python como imagen base. Debian 11 (bullseye) terminó su
# soporte LTS en agosto de 2026 y sus paquetes de seguridad ya no se sirven desde
# deb.debian.org, así que la construcción fallaba con 404; bookworm sigue soportado.
FROM python:3.11-slim-bookworm

# Define el directorio de trabajo dentro del contenedor
WORKDIR /MoneyPrinterTurbo

# Pone los permisos del directorio /MoneyPrinterTurbo a 777
RUN chmod 777 /MoneyPrinterTurbo

ENV PYTHONPATH="/MoneyPrinterTurbo"

# Los usuarios locales siguen usando por defecto primero los espejos de China; al publicar la imagen en GHCR desde GitHub Actions se usa default,
# para que los runners fuera de China no tarden demasiado en acceder a los espejos chinos y la publicación de la imagen no se atasque.
ARG DOCKER_BUILD_MIRROR=china
ARG PIP_USE_OFFICIAL=0

# La instalación de dependencias del sistema debe cumplir dos cosas a la vez: mantener en China la posibilidad de volver a otros espejos y, si fallan
# todos los espejos, hacer fallar de inmediato la construcción de Docker. El sleep que ejecutaba al final el bucle antiguo siempre devolvía 0,
# así que se generaba una imagen inservible aunque no se hubieran instalado git/ffmpeg. Aquí "escribir la fuente de software", "instalar"
# y "tres reintentos" se separan en funciones de shell con límites claros, y el valor de retorno de la función decide si se continúa.
# Todas las fuentes de software usan HTTPS, para que algunas redes no bloqueen directamente las solicitudes HTTP en claro.
RUN set -u; \
    write_debian_sources() { \
        main_url="$1"; \
        security_url="$2"; \
        printf 'deb %s bookworm main\ndeb %s bookworm-updates main\ndeb %s bookworm-security main\n' \
            "$main_url" "$main_url" "$security_url" > /etc/apt/sources.list; \
        rm -f /etc/apt/sources.list.d/debian.sources; \
        rm -rf /var/lib/apt/lists/*; \
    }; \
    install_system_dependencies() { \
        apt-get update && \
        apt-get install -y --no-install-recommends git ffmpeg; \
    }; \
    retry_system_dependencies() { \
        attempt=1; \
        while [ "$attempt" -le 3 ]; do \
            echo "Intento $attempt: instalando dependencias del sistema"; \
            if install_system_dependencies; then \
                return 0; \
            fi; \
            echo "El intento $attempt falló" >&2; \
            if [ "$attempt" -lt 3 ]; then \
                echo "Reintentando en 5 segundos..." >&2; \
                sleep 5; \
            fi; \
            attempt=$((attempt + 1)); \
        done; \
        return 1; \
    }; \
    if [ "$DOCKER_BUILD_MIRROR" = "china" ]; then \
        write_debian_sources \
            "https://mirrors.aliyun.com/debian" \
            "https://mirrors.aliyun.com/debian-security"; \
        if ! retry_system_dependencies; then \
            echo "Falló el espejo de Aliyun, se cambia al espejo de Tsinghua" >&2; \
            write_debian_sources \
                "https://mirrors.tuna.tsinghua.edu.cn/debian" \
                "https://mirrors.tuna.tsinghua.edu.cn/debian-security"; \
            if ! install_system_dependencies; then \
                echo "Falló el espejo de Tsinghua, se cambia al espejo de Debian por defecto" >&2; \
                write_debian_sources \
                    "https://deb.debian.org/debian" \
                    "https://deb.debian.org/debian-security"; \
                if ! install_system_dependencies; then \
                    echo "No se pudieron instalar las dependencias del sistema desde ningún espejo configurado" >&2; \
                    exit 1; \
                fi; \
            fi; \
        fi; \
    else \
        echo "Usando los espejos de Debian por defecto"; \
        write_debian_sources \
            "https://deb.debian.org/debian" \
            "https://deb.debian.org/debian-security"; \
        if ! retry_system_dependencies; then \
            echo "No se pudieron instalar las dependencias del sistema desde el espejo de Debian por defecto" >&2; \
            exit 1; \
        fi; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# Copia primero solo requirements.txt para aprovechar la caché de Docker
COPY requirements.txt ./

# En local se usa por defecto primero el espejo de PyPI de China; la publicación en GHCR usa el PyPI oficial, para que los runners fuera de China no vayan lentos por el acceso transfronterizo al espejo.
RUN if [ "$PIP_USE_OFFICIAL" = "1" ]; then \
        pip install --no-cache-dir --retries 3 --timeout 60 -r requirements.txt; \
    else \
        pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --retries 3 --timeout 60 -r requirements.txt || \
        pip install --no-cache-dir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/ --trusted-host mirrors.tuna.tsinghua.edu.cn --retries 3 --timeout 60 -r requirements.txt || \
        pip install --no-cache-dir --retries 3 --timeout 60 -r requirements.txt; \
    fi

# Ahora copia el resto del código a la imagen
COPY . .

# Expone el puerto en el que se ejecuta la aplicación
EXPOSE 8501

# Dentro del contenedor hay que escuchar en 0.0.0.0; el host sigue limitándolo a 127.0.0.1 con el mapeo de puertos de Docker.
# browser.serverAddress solo decide la dirección de acceso que muestra el navegador y no sustituye a server.address.
CMD ["streamlit", "run", "./webui/Main.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.serverAddress=127.0.0.1", "--server.enableCORS=True", "--browser.gatherUsageStats=False", "--client.toolbarMode=minimal", "--logger.hideWelcomeMessage=True", "--server.showEmailPrompt=False"]

# 1. Construye la imagen de Docker con el siguiente comando
# docker build -t moneyprinterturbo .

# 2. Ejecuta el contenedor de Docker con el siguiente comando
## En Linux o macOS:
# docker run -v $(pwd)/config.toml:/MoneyPrinterTurbo/config.toml -v $(pwd)/storage:/MoneyPrinterTurbo/storage -p 127.0.0.1:8501:8501 moneyprinterturbo
## En Windows:
# docker run -v ${PWD}/config.toml:/MoneyPrinterTurbo/config.toml -v ${PWD}/storage:/MoneyPrinterTurbo/storage -p 127.0.0.1:8501:8501 moneyprinterturbo
