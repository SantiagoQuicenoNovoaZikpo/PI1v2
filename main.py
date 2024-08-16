import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Cargar el DataFrame desde el archivo Parquet
df = pd.read_parquet('df.parquet', engine='pyarrow')

# Crear la instancia de la aplicación FastAPI
app = FastAPI()

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

@app.get("/cantidad_filmaciones_mes")
def cantidad_filmaciones_mes(mes: str):
    mes_ingles = month_translation.get(mes.lower())
    
    if not mes_ingles:
        raise HTTPException(status_code=400, detail="Mes no válido, por favor escríbalo en español.")
    
    count = df[df['release_date'].dt.strftime('%B') == mes_ingles].shape[0]
    
    return {
        f"{count} fue la cantidad de películas estrenadas en el mes de {mes.capitalize()}"
    }

@app.get("/cantidad_filmaciones_dia")
def cantidad_filmaciones_dia(dia: str):
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    dias_espanol_a_ingles = {
        'Lunes': 'Monday',
        'Martes': 'Tuesday',
        'Miércoles': 'Wednesday',
        'Jueves': 'Thursday',
        'Viernes': 'Friday',
        'Sábado': 'Saturday',
        'Domingo': 'Sunday'
    }

    dia_ingles = dias_espanol_a_ingles.get(dia.capitalize(), None)

    if not dia_ingles:
        return f"Día '{dia}' no es válido. Usa un nombre de día en Español."

    df['day_of_week'] = df['release_date'].dt.day_name()
    cantidad = df[df['day_of_week'] == dia_ingles].shape[0]

    return f"{cantidad} cantidad de películas fueron estrenadas en los días {dia}."

@app.get("/score_titulo")
def score_titulo(titulo_de_la_filmación: str):
    pelicula = df[df['title'].str.contains(titulo_de_la_filmación, case=False, na=False)]

    if pelicula.empty:
        return f"No se encontró ninguna película con el título '{titulo_de_la_filmación}'."

    pelicula_info = pelicula.iloc[0]
    titulo = pelicula_info['title']
    año_estreno = pelicula_info['release_year']
    score = pelicula_info.get('vote_average', 'No disponible')

    return f"La película '{titulo}' fue estrenada en el año {año_estreno} con un puntaje de {score}."

@app.get("/votos_titulo")
def votos_titulo(titulo_de_la_filmación: str):
    pelicula = df[df['title'].str.contains(titulo_de_la_filmación, case=False, na=False)]

    if pelicula.empty:
        return f"No se encontró ninguna película con el título '{titulo_de_la_filmación}'."

    pelicula_info = pelicula.iloc[0]
    titulo = pelicula_info['title']
    cantidad_votos = pelicula_info['vote_count']
    promedio_votos = pelicula_info['vote_average']
    
    if cantidad_votos < 2000:
        return f"La película '{titulo}' no tiene al menos 2000 valoraciones. Solo cuenta con {cantidad_votos} valoraciones."

    return f"La película '{titulo}' cuenta con un total de {cantidad_votos} valoraciones, con un promedio de {promedio_votos:.2f}."

@app.get("/get_actor")
def get_actor(nombre_actor: str):
    actor_movies = df[df['name'] == nombre_actor]
    
    if actor_movies.empty:
        return f"No se encontraron películas para el actor '{nombre_actor}'."
    
    num_peliculas = actor_movies.shape[0]
    retorno_total = actor_movies['revenue'].sum()
    promedio_retorno = actor_movies['revenue'].mean()
    
    return (f"El actor {nombre_actor} ha participado en {num_peliculas} filmaciones, "
            f"ha conseguido un retorno de {retorno_total} con un promedio de {promedio_retorno:.2f} por filmación.")

@app.get("/get_director")
def get_director(nombre_director: str):
    directores_peliculas = []
    
    for i, row in df.iterrows():
        director_actual = row['crew']
        
        if director_actual == nombre_director:
            retorno = row['revenue'] / row['budget'] if row['budget'] != 0 else 0
            ganancia = row['revenue'] - row['budget']
            costo = row['budget']
            
            pelicula = {
                'titulo': row['title'],
                'fecha_lanzamiento': row['release_date'],
                'retorno': retorno,
                'costo': costo,
                'ganancia': ganancia,
                'calificacion': row.get('vote_average', None)
            }
            directores_peliculas.append(pelicula)
    
    resultado = pd.DataFrame(directores_peliculas)
    
    if 'calificacion' in resultado.columns:
        calificaciones = resultado['calificacion'].apply(pd.to_numeric, errors='coerce')
        promedio_calificaciones = calificaciones.mean() if not calificaciones.empty else 0
    else:
        promedio_calificaciones = 0
    
    promedio_director = pd.DataFrame([{
        'titulo': nombre_director,
        'fecha_lanzamiento': '',
        'retorno': '',
        'costo': '',
        'ganancia': '',
        'calificacion': promedio_calificaciones
    }])
    
    resultado = pd.concat([promedio_director, resultado], ignore_index=True)
    
    return resultado



# Vectorizar la columna de títulos para calcular la similitud
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['title'])

# Calcular la matriz de similitud de coseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


@app.get("/recomendacion")
def recomendacion(titulo):
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
