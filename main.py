import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Cargar el DataFrame desde el archivo Parquet
#df = pd.read_parquet('df.parquet', engine='pyarrow')


# Crear la instancia de la aplicación FastAPI
app = FastAPI()

@app.get("/cantidad_filmaciones_mes")
def cantidad_filmaciones_mes(mes: str):

    # Diccionario para traducir los nombres de los meses en español a inglés
    month_translation = {
    'enero': 'January',
    'febrero': 'February',
    'marzo': 'March',
    'abril': 'April',
    'mayo': 'May',
    'junio': 'June',
    'julio': 'July',
    'agosto': 'August',
    'septiembre': 'September',
    'octubre': 'October',
    'noviembre': 'November',
    'diciembre': 'December'
}
    # Convertir el mes en español a inglés
    mes_ingles = month_translation.get(mes.lower())
    
    # Cargar el DataFrame desde el archivo Parquet
    df = pd.read_parquet('df.parquet', engine='pyarrow')

    if not mes_ingles:
        raise HTTPException(status_code=400, detail="Mes no válido, porfavor escribalo en espa;ol  ")
    
    # Filtrar el DataFrame para contar las películas estrenadas en el mes indicado
    count = df[df['release_date'].dt.strftime('%B') == mes_ingles].shape[0]
    
    return {
        f"{count} fue la cantidad de películas fueron estrenadas en el mes de {mes.capitalize()}"
    }


@app.get("/cantidad_filmaciones_dia")
def cantidad_filmaciones_dia(dia: str) -> str:

    # Cargar el DataFrame desde el archivo Parquet
    df = pd.read_parquet('df.parquet', engine='pyarrow')
    
    # Asegurarse de que la columna 'release_date' sea de tipo fecha
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Convertir el día del mes a número (por ejemplo, 'Lunes' a 'Monday')
    dias_espanol_a_ingles = {
        'Lunes': 'Monday',
        'Martes': 'Tuesday',
        'Miércoles': 'Wednesday',
        'Jueves': 'Thursday',
        'Viernes': 'Friday',
        'Sábado': 'Saturday',
        'Domingo': 'Sunday'
    }

    # Obtener el nombre del día en inglés
    dia_ingles = dias_espanol_a_ingles.get(dia.capitalize(), None)

    if not dia_ingles:
        return f"Día '{dia}' no es válido. Usa un nombre de día en Español."

    # Filtrar las películas estrenadas en el día específico
    df['day_of_week'] = df['release_date'].dt.day_name()
    peliculas_dia = df[df['day_of_week'] == dia_ingles]

    # Contar el número de películas
    cantidad = peliculas_dia.shape[0]

    return f"{cantidad} cantidad de películas fueron estrenadas en los días {dia}."

@app.get("/score_titulo")
def score_titulo(titulo_de_la_filmación: str) -> str:

    # Cargar el DataFrame desde el archivo Parquet
    df = pd.read_parquet('df.parquet', engine='pyarrow')

    # Filtrar el DataFrame para encontrar la película por título
    pelicula = df[df['title'].str.contains(titulo_de_la_filmación, case=False, na=False)]

    # Verificar si se encontró la película
    if pelicula.empty:
        return f"No se encontró ninguna película con el título '{titulo_de_la_filmación}'."

    # Extraer información
    # Suponiendo que solo hay una película con ese título, si hay múltiples, ajustar según sea necesario
    pelicula_info = pelicula.iloc[0]
    titulo = pelicula_info['title']
    año_estreno = pelicula_info['release_year']
    score = pelicula_info.get('vote_average', 'No disponible')  # Ajustar si el nombre de la columna es diferente

    return f"La película '{titulo}' fue estrenada en el año {año_estreno} con un puntaje de {score}."

@app.get("/votos_titulo")
def votos_titulo(titulo_de_la_filmación: str) -> str:

    # Cargar el DataFrame desde el archivo Parquet
    df = pd.read_parquet('df.parquet', engine='pyarrow')

    # Filtrar el DataFrame para encontrar la película por título
    pelicula = df[df['title'].str.contains(titulo_de_la_filmación, case=False, na=False)]

    # Verificar si se encontró la película
    if pelicula.empty:
        return f"No se encontró ninguna película con el título '{titulo_de_la_filmación}'."

    # Extraer información
    pelicula_info = pelicula.iloc[0]
    titulo = pelicula_info['title']
    cantidad_votos = pelicula_info['vote_count']
    promedio_votos = pelicula_info['vote_average']
    
    # Verificar si la cantidad de votos es menor a 2000
    if cantidad_votos < 2000:
        return f"La película '{titulo}' no tiene al menos 2000 valoraciones. Solo cuenta con {cantidad_votos} valoraciones."

    return (f"La película '{titulo}' cuenta con un total de {cantidad_votos} valoraciones, con un promedio de {promedio_votos:.2f}.")

@app.get("/get_actor")
def get_actor(nombre_actor: str) -> str:

    # Cargar el DataFrame desde el archivo Parquet
    df = pd.read_parquet('df.parquet', engine='pyarrow')
    
    # Filtrar las películas en las que el actor esté presente
    actor_movies = df[df['name'] == nombre_actor]
    
    # Verificar si se encontraron películas para el actor
    if actor_movies.empty:
        return f"No se encontraron películas para el actor '{nombre_actor}'."
    
    # Calcular la cantidad de películas, el retorno total y el promedio de retorno
    num_peliculas = actor_movies.shape[0]
    retorno_total = actor_movies['revenue'].sum()
    promedio_retorno = actor_movies['revenue'].mean()
    
    return (f"El actor {nombre_actor} ha participado en {num_peliculas} filmaciones, "
            f"ha conseguido un retorno de {retorno_total} con un promedio de {promedio_retorno:.2f} por filmación.")

@app.get("/get_director")
def get_director(nombre_director):
    directores_peliculas = []

    # Cargar el DataFrame desde el archivo Parquet
    df = pd.read_parquet('df.parquet', engine='pyarrow')
    
    for i, row in df.iterrows():
        # Obtener el nombre del director directamente de la columna 'Director'
        director_actual = row['Director']

        # Verificar si director_actual es una lista
        if isinstance(director_actual, list):
            if nombre_director in director_actual:
                # Calculamos el retorno, ganancia y costo
                retorno = row['revenue'] / row['budget'] if row['budget'] != 0 else 0
                ganancia = row['revenue'] - row['budget']
                costo = row['budget']
                
                # Añadimos los detalles de la película a la lista
                pelicula = {
                    'titulo': row['title'],
                    'fecha_lanzamiento': row['release_date'],
                    'retorno': retorno,
                    'costo': costo,
                    'ganancia': ganancia,
                    'calificacion': row.get('vote_average', None)  # Añadimos la calificación de la película
                }
                directores_peliculas.append(pelicula)
        elif isinstance(director_actual, str):
            if director_actual == nombre_director:
                # Calculamos el retorno, ganancia y costo
                retorno = row['revenue'] / row['budget'] if row['budget'] != 0 else 0
                ganancia = row['revenue'] - row['budget']
                costo = row['budget']
                
                # Añadimos los detalles de la película a la lista
                pelicula = {
                    'titulo': row['title'],
                    'fecha_lanzamiento': row['release_date'],
                    'retorno': retorno,
                    'costo': costo,
                    'ganancia': ganancia,
                    'calificacion': row.get('vote_average', None)  # Añadimos la calificación de la película
                }
                directores_peliculas.append(pelicula)
    
    # Creamos un DataFrame con los resultados
    resultado1 = pd.DataFrame(directores_peliculas)
    
    # Asegúrate de que la columna 'calificacion' existe y convierte a numérico
    if 'calificacion' in resultado1.columns:
        calificaciones = resultado1['calificacion'].apply(pd.to_numeric, errors='coerce')
        promedio_calificaciones = calificaciones.mean() if not calificaciones.empty else 0
    else:
        promedio_calificaciones = 0
    

    # Añadimos una fila con el promedio de calificaciones del director
    promedio_director = pd.DataFrame([{
        'titulo': nombre_director,
        'fecha_lanzamiento': '',
        'retorno': '',
        'costo': '',
        'ganancia': '',
        'calificacion': promedio_calificaciones
    }])
    
    # Concatenamos los resultados
    resultado = pd.concat([promedio_director,resultado1], ignore_index=True)
    
    return resultado



@app.get("/recomendacion")
def recomendacion(titulo):

    # Cargar el DataFrame desde el archivo Parquet
    df = pd.read_parquet('df.parquet', engine='pyarrow')

    # Asegurarse de que el título esté en el DataFrame
    if titulo not in df['title'].values:
        return f"La película '{titulo}' no está en el dataset."

    # Crear una serie que mapea títulos de películas a sus índices
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    # Obtener el índice de la película que coincide con el título
    idx = indices.get(titulo)
    
    # Si hay múltiples índices, escoger el primero (puedes ajustar esto si lo prefieres)
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    
    # Vectorizar la columna de títulos para calcular la similitud
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dfm['title'])

    # Calcular la matriz de similitud de coseno
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Calcular las puntuaciones de similitud
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordenar las películas por similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Seleccionar las 5 películas más similares, excluyendo la película original
    sim_scores = sim_scores[1:6]
    
    # Obtener los índices de las películas
    movie_indices = [i[0] for i in sim_scores]
    
    # Devolver los títulos de las películas similares
    return df['title'].iloc[movie_indices].tolist()
