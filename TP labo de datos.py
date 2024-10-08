#%% ==========================================================================================
# Importamos librerias
import pandas as pd
from inline_sql import sql, sql_val

#%% ==========================================================================================
# Importamos los CSV 

carpeta = "C:/Users/franc/Documents/Facultad/Laboratorio de datos/"
           

lista_secciones = pd.read_csv(carpeta+"lista-secciones.csv")
lista_secciones 

lista_sedes = pd.read_csv(carpeta+"lista-sedes.csv")
lista_sedes 

lista_sedes_datos = pd.read_csv(carpeta+"lista-sedes-datos.csv", on_bad_lines='skip')
lista_sedes_datos 

migraciones = pd.read_csv(carpeta+"migraciones.csv")
migraciones 

#%% ==========================================================================================
# Limpieza de datos

# Hay un Null en sede_id
lista_secciones.info()

# Lo saco porque es la columna mas importante de esta tabla
consultaSQL = """
               SELECT sede_id, sede_desc_castellano AS sede, tipo_seccion  
               FROM lista_secciones
               WHERE sede_id IS NOT NULL;
              """

lista_secciones_corregida = sql^ consultaSQL
print(lista_secciones_corregida) 

# Copruebo que ya no hay NULLS
lista_secciones_corregida.info()

# Compruebo que no todas las filas de tipo_seccion son 'seccion' porque de lo contrario seria informacion irrelevante 
sin_seccion = lista_secciones_corregida['tipo_seccion'] != 'Seccion'
lista_secciones_corregida[sin_seccion]

#%% ==========================================================================================

# No hay NULLS
lista_sedes.info()

consultaSQL = """
               SELECT sede_id, sede_desc_castellano AS sede, pais_castellano AS pais, ciudad_castellano AS ciudad, estado, sede_tipo, pais_iso_3 AS iso_3  
               FROM lista_sedes;
              """

lista_sedes_corregida = sql^ consultaSQL
print(lista_sedes_corregida)

#%% ==========================================================================================

# Las columnas con NULLS son irrelevantes para el analisis 
lista_sedes_datos.info()

consultaSQL = """
               SELECT sede_id, sede_desc_castellano AS sede, pais_castellano AS pais, ciudad_castellano AS ciudad, 
               redes_sociales, pais_iso_3 AS iso_3, region_geografica      
               FROM lista_sedes_datos;
              """

lista_sede_datos_corregida = sql^ consultaSQL
print(lista_sede_datos_corregida)

#%% ==========================================================================================

# Limpieza de datos en la tabla de migraciones, cambie el nombre de las columnas porque me daban problemas
# Quite los nulls.
consultaSQL = """
               SELECT CountryOriginName AS pais_de_origen, CountryOriginCode AS codigo_pais_origen, 
               CountryDestName AS pais_destino, CountryDestCode AS codigo_pais_destino,
               sesentas, setentas, ochentas, noventas, dosmil
               FROM migraciones
               WHERE sesentas IS NOT NULL AND
                     setentas IS NOT NULL AND
                     ochentas IS NOT NULL AND
                     noventas IS NOT NULL AND
                     dosmil IS NOT NULL;
              """
migraciones_sin_nulls = sql^ consultaSQL
# Ademas de los nulls, hay muchas filas con el valor '..' para sacarlo tuve que convertir las columnas a string y despues nuevamente a enteros
consultaSQL = """
               SELECT pais_de_origen, codigo_pais_origen, 
                 pais_destino, codigo_pais_destino,
                 CAST(sesentas AS INT) AS sesentas,
                 CAST(setentas AS INT) AS setentas,
                 CAST(ochentas AS INT) AS ochentas,
                 CAST(noventas AS INT) AS noventas,
                 CAST(dosmil AS INT) AS dosmil
               FROM (
                 SELECT pais_de_origen, codigo_pais_origen, 
                  pais_destino, codigo_pais_destino,
                  CAST(sesentas AS STRING) AS sesentas,
                  CAST(setentas AS STRING) AS setentas,
                  CAST(ochentas AS STRING) AS ochentas,
                  CAST(noventas AS STRING) AS noventas,
                  CAST(dosmil AS STRING) AS dosmil
                  FROM migraciones_sin_nulls
               ) AS subquery
               WHERE 
                 sesentas != '..' AND
                 setentas != '..' AND
                 ochentas != '..' AND
                 noventas != '..' AND
                 dosmil != '..';
              """
migraciones_filtrada = sql^ consultaSQL
migraciones_filtrada

# Y ahora si puedo sacar todas aquellas donde las cinco filas son cero
consultaSQL = """
               SELECT pais_de_origen, codigo_pais_origen, 
               pais_destino, codigo_pais_destino,
               sesentas AS "1960", setentas AS "1970", ochentas AS "1980", noventas AS "1990", dosmil AS "2000"
               FROM migraciones_filtrada
               WHERE sesentas != 0 AND setentas != 0 AND ochentas != 0 AND noventas != 0 AND dosmil != 0;
              """
migraciones_corregida = sql^ consultaSQL
migraciones_corregida

#%% ==========================================================================================
 
# Punto h) 

# I)
# Creo una tabla con la cantidad de secciones promedio por pais
consultaSQL = """
                 SELECT AVG(cantidad_de_secciones) AS secciones_promedio,
                 iso_3, 
                 FROM (
                   SELECT COUNT(tipo_seccion) AS cantidad_de_secciones,
                   lista_secciones_corregida.sede_id,
                   iso_3 
                   FROM lista_secciones_corregida
                   INNER JOIN lista_sede_datos_corregida
                   ON lista_secciones_corregida.sede_id = lista_secciones_corregida.sede_id
                   WHERE tipo_seccion = 'Seccion'
                   GROUP BY lista_secciones_corregida.sede_id, iso_3
                 )
                 GROUP BY iso_3
              """

cantidad_de_secciones_promedio = sql^ consultaSQL
cantidad_de_secciones_promedio

# Creo una tabla con la cantidad de secciones promedio, cantidad de sedes, ordenados por pais
consultaSQL = """
                 SELECT cantidad_de_secciones_promedio.secciones_promedio,
                 cantidad_de_secciones_promedio.iso_3,
                 sedes
                 FROM cantidad_de_secciones_promedio
                 INNER JOIN (
                   SELECT 
                   iso_3,  
                   COUNT(*) AS sedes
                   FROM lista_sede_datos_corregida
                   GROUP BY iso_3
                 ) AS cantidad_de_sedes
                 ON cantidad_de_secciones_promedio.iso_3 = cantidad_de_sedes.iso_3;
              """

secciones_promedio_sedes = sql^ consultaSQL
secciones_promedio_sedes

# Migraciones en el año 2000
consultaSQL = """
                 SELECT (cantidad_de_inmigracion - cantidad_de_emigracion) AS flujo_migratorio_neto,
                 codigo_pais_origen AS iso_3,
                 pais_de_origen AS Pais
                 FROM (
                     SELECT pais_de_origen,
                     codigo_pais_origen,
                     SUM("2000") AS cantidad_de_emigracion
                     FROM migraciones_corregida
                     GROUP BY pais_de_origen, codigo_pais_origen
                 ) AS emigracion
                 INNER JOIN (
                     SELECT pais_destino,
                     codigo_pais_destino,
                     SUM("2000") AS cantidad_de_inmigracion
                     FROM migraciones_corregida
                     GROUP BY pais_destino, codigo_pais_destino
                 ) AS inmigracion
                 ON emigracion.codigo_pais_origen = inmigracion.codigo_pais_destino;
              """

flujo_migratorio = sql^ consultaSQL
flujo_migratorio

# Reporte final 
consultaSQL = """
                 SELECT Pais,
                 sedes,
                 secciones_promedio,
                 flujo_migratorio_neto,
                 FROM secciones_promedio_sedes
                 INNER JOIN flujo_migratorio
                 ON secciones_promedio_sedes.iso_3 = flujo_migratorio.iso_3
                 ORDER BY sedes DESC, Pais ASC;
              """

reporte = sql^ consultaSQL
reporte

#%% ==========================================================================================
#ii)

# cuento sin repetir los paises con sedes en argentina
consultaSQL = """
SELECT region_geografica,
       COUNT(DISTINCT pais) AS paises_con_sedes_argentinas,
       FROM lista_sede_datos_corregida
       GROUP BY region_geografica

"""
paises_sedes = sql^ consultaSQL 


#separo en una tabla los paises donde argentina tiene sedes
# asi lo uso para hacer el promedio que me piden 

consultaSQL = """
SELECT DISTINCT iso_3, region_geografica
FROM lista_sede_datos_corregida
"""
pais_destin = sql^ consultaSQL

#hago un inner join con los paises donde argentina tiene sedes y el pais de destino en migraciones
# Ademas filtro las columnas donde el pais de origen sea argentina
consultaSQL = """
SELECT pais_de_origen, codigo_pais_destino, "2000", region_geografica
FROM migraciones_corregida
INNER JOIN pais_destin
ON migraciones_corregida.codigo_pais_destino = pais_destin.iso_3
WHERE pais_de_origen = 'Argentina'
"""

flujo_arg= sql^ consultaSQL

# calculo el promedio, primero cuento la cantidad de paises a la que arribaron desde arg
# despues hago el promedio sobre esa suma
# Agregue un join para hacer el flujo
consultaSQL = """ 
SELECT region_geografica,
AVG(inmigracion - emigracion) AS promedio_flujo_con_arg  
FROM (
      SELECT region_geografica,
             codigo_pais_destino, SUM("2000") AS emigracion
      FROM flujo_arg
      GROUP BY region_geografica, codigo_pais_destino
      ) AS em
INNER JOIN (
      SELECT codigo_pais_origen, SUM("2000") AS inmigracion
      FROM migraciones_corregida
      INNER JOIN pais_destin
      ON migraciones_corregida.codigo_pais_origen = pais_destin.iso_3
      WHERE pais_destino = 'Argentina'
      GROUP BY codigo_pais_origen
) AS inm 
ON em.codigo_pais_destino = inm.codigo_pais_origen    
GROUP BY region_geografica
"""

promedio_flujo= sql^ consultaSQL


# uno las tablas de los paises con sede en arg y del promedio
consultaSQL = """
SELECT paises_sedes.region_geografica,
       paises_con_sedes_argentinas,
       promedio_flujo_con_arg
FROM paises_sedes
INNER JOIN promedio_flujo
ON paises_sedes.region_geografica = promedio_flujo.region_geografica
GROUP BY paises_sedes.region_geografica, paises_sedes.paises_con_sedes_argentinas, promedio_flujo.promedio_flujo_con_arg
ORDER BY promedio_flujo_con_arg DESC

"""
reporte2 = sql^ consultaSQL 
reporte2 

#%% ==========================================================================================
# iii)

# Hay valores nulos en redes_sociales
lista_sede_datos_corregida.info()

# Los saco porque no suman nada en este reporte
consultaSQL = """
               SELECT *
               FROM lista_sede_datos_corregida
               WHERE redes_sociales IS NOT NULL;
              """

redes_sin_nulls = sql^ consultaSQL
redes_sin_nulls

# Creo una nueva tabla con columnas para cada red social, si la sede tiene dicha red social se asigna un 1, de lo contrario 0

redes = redes_sin_nulls

redes['facebook'] = redes_sin_nulls['redes_sociales'].str.contains('facebook', case=False).astype(int)

redes['instagram'] = redes_sin_nulls['redes_sociales'].str.contains('instagram', case=False).astype(int)

redes['twitter'] = redes_sin_nulls['redes_sociales'].str.contains('twitter', case=False).astype(int)

redes['youtube'] = redes_sin_nulls['redes_sociales'].str.contains('youtube', case=False).astype(int)

redes['cantidad_de_redes'] = redes['facebook'] + redes['instagram'] + redes['twitter'] + redes['youtube']

# Creo una tabla que indica que cantidad de redes sociales que tiene cada pais
consultaSQL = """
               SELECT iso_3,
               pais_de_origen AS pais,
               MAX(cantidad_de_redes) AS cantidad_de_redes
               FROM redes
               INNER JOIN migraciones_corregida
               ON iso_3 = codigo_pais_origen
               GROUP BY iso_3, pais_de_origen;
              """

reporte3 = sql^ consultaSQL
reporte3