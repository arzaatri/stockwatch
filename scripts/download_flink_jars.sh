#!/usr/bin/env bash
# Downloads the Flink Kafka connector JAR needed by streaming/flink_job.py's KafkaSource/KafkaSink.
# Version must be compatible with the installed `apache-flink` (pyflink) version - see pyproject.toml.
set -euo pipefail

JAR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/jars"
CONNECTOR_VERSION="5.0.0-2.2"
JAR_NAME="flink-sql-connector-kafka-${CONNECTOR_VERSION}.jar"
URL="https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/${CONNECTOR_VERSION}/${JAR_NAME}"

mkdir -p "${JAR_DIR}"
if [ -f "${JAR_DIR}/${JAR_NAME}" ]; then
  echo "Already have ${JAR_NAME}"
else
  echo "Downloading ${JAR_NAME}..."
  curl -sSL -o "${JAR_DIR}/${JAR_NAME}" "${URL}"
  echo "Saved to ${JAR_DIR}/${JAR_NAME}"
fi
