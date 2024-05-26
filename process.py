
from geopy.geocoders import Nominatim
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, mapping
import osmnx as ox
import h3
import numpy as np
import googlemaps


def get_coordinates(endereco: str) -> tuple:
    """
    Retorna as coordenadas geográficas de um endereço.

    Args:
        endereco (str): O endereço a ser pesquisado.

    Returns:
        Um tuple com as coordenadas geográficas (latitude, longitude).
    """

    geolocator = Nominatim(user_agent="test_app")
    location = geolocator.geocode(endereco)

    if location:
        return location.latitude, location.longitude
    else:
        # Isso pode dar problemas quando transformar para uma objeto geográfico
        return None


def create_geo_address(enderecos: list) -> gpd.GeoDataFrame:
    """
    Cria um GeoDataFrame a partir de uma lista de endereços.

    Args:
        enderecos (list): Uma lista de endereços.

    Returns:
        Um GeoDataFrame com as coordenadas geográficas dos endereços.
    """

    coordenadas = [get_coordinates(i) for i in enderecos]
    geometry = [
        Point(longitude, latitude) for latitude, longitude in coordenadas
        ]
    enderecos_gpd = gpd.GeoDataFrame(
        {'Endereços': enderecos, 'geometry': geometry}, crs='EPSG:4326'
        )
    enderecos_gpd = enderecos_gpd.reset_index()
    return enderecos_gpd


def create_geo_objects_city(place_name: str) -> gpd.GeoDataFrame:
    """
    Cria arquivos geográficos através do nome de uma cidade indicada

    Args:
        place_name (str): Nome da cidade.

    Returns:
        geodataframes nodes_proj, edges_proj e area
    """
    graph = ox.graph_from_place(place_name)
    nodes_proj, edges_proj = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    area = ox.geocode_to_gdf(place_name)
    # Criar uma cópia do GeoDataFrame original para evitar modificar o original
    # buffered_area = area.copy()
    # O argumento representa a distância do
    # #buffer em metros (2 km = 2000 metros)
    # buffered_area['geometry'] = area.buffer(2000)
    # buffered_area.crs = area.crs

    return nodes_proj, edges_proj, area  # buffered_area


def give_facilities_googlemaps(key: str, latitude: float, longitude: float,
                               radius: int, type: str) -> gpd.GeoDataFrame:
    """
    Buscas através de uma API do Google Maps dentro do raio especificado de
    uma coordenada um tipo de estabelicimento solicitado
    Args:
        key (str): Chave de acesso a API
        latitude (float): latitude geográfica
        longitude (float): longitude geográfica
        radius (int): Raio de busca em metros
        type (str): Tipo de estabelecimento a ser buscado
        (https://developers.google.com/maps/documentation/places/web-service/place-types)

    Returns:
        geodataframe com a lista de estabelecimento em detalhes
    """
    gmaps = googlemaps.Client(key=key)
    places = gmaps.places_nearby(
        location=(latitude, longitude),  # Coordenadas do ponto central
        radius=radius,  # Raio em metros
        type=type
    )

    name = [place['name'] for place in places['results']]
    vicinity = [place['vicinity'] for place in places['results']]
    types = [place['types'] for place in places['results']]
    place_id = [place['place_id'] for place in places['results']]
    geometry = [
        Point(geo['geometry']['location']['lng'],
              geo['geometry']['location']['lat']) for geo in places['results']
        ]

    facilities_gpd = gpd.GeoDataFrame({'main_type': [type for i in name],
                                       'Nome': name,
                                       'Endereco': vicinity,
                                       'types': types,
                                       'place_id': place_id,
                                       'geometry': geometry}, crs='EPSG:4326')
    return facilities_gpd


def facilities_for_locates_range(
        types: list, df_gpd: gpd.GeoDataFrame,
        index_gpd: str, key: str,
        geometry_col: str = 'geometry') -> gpd.GeoDataFrame:
    """
    Busca estabelecimentos de uma lista de tipos em um raio de 1km para cada
    ponto geográfico do geodataframe passado, e retorna um geodataframe com
    todas as informações coletadas. Os estabelecimentos são buscados através
    da API do Google Maps.

    Args:
        types (list): Uma lista de strings com os tipos de estabelecimentos a
        serem buscados.
        df_gpd (gpd.GeoDataFrame): Um GeoDataFrame com os pontos geográficos
        onde serão feitas as buscas.
        index_gpd (str): O nome da coluna do GeoDataFrame que contém os índices
        dos pontos geográficos.
        key (str): A chave de acesso à API do Google Maps.
        geometry_col(str): O nome da coluna do GeoDataFrame que contém os
        objetos geométricos.

    Returns:
        Um GeoDataFrame com todas as informações coletadas sobre os
        estabelecimentos encontrados.
    """

    full_places = gpd.GeoDataFrame()
    for type in types:
        print(f"locate {type}...")
        for place_index in df_gpd[index_gpd]:
            place_geometry_y = df_gpd[df_gpd[index_gpd] == place_index][geometry_col].y
            place_geometry_x = df_gpd[df_gpd[index_gpd] == place_index][geometry_col].x
            facilities_gpd = give_facilities_googlemaps(key=key,
                                                        latitude=place_geometry_y,
                                                        longitude=place_geometry_x,
                                                        radius=1000,
                                                        type=type)
            facilities_gpd['place_index'] = place_index
            full_places = pd.concat([full_places, facilities_gpd])
    full_places = full_places.drop_duplicates(subset='place_id')

    return full_places


def make_hexagons(area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Função para gerar hexágonos de diferentes aberturas a partir da geometria
    de uma area.

    Eu adaptei essa função do artigo do Alvaro Matsuda, aqui está ele
    (https://kazumatsuda.medium.com/spatial-clustering-fa2ea5d035a3)

    Parâmetros:
        area (gpd.GeoDataFrame): GeoDataFrame com a geometria.

    Retorno:
        GeoDataFrame com os centroides dos hexágonos de tamanho final.
    """

    # Lista vazia a ser preenchida com índices hexágonos
    hex_list = []
    # Tamanho inicial da abertura dos hexágonos (int).
    initial_aperture_size = 9
    # Tamanho final da abertura dos hexágonos (int).
    final_aperture_size = 8
    # Iterar sobre geometria da cidade para obter índices hexágonos
    for n, g in enumerate(area['geometry'].explode(ignore_index=True)):

        # Obtenha GeoJson da geometria
        temp = mapping(g)

        # Obtenha coordenadas de geometria do GeoJson
        temp['coordinates'] = [
            [[j[1], j[0]] for j in i] for i in temp['coordinates']
            ]

        # Preenche o polígono com hexágonos contidos na estrutura de dados
        # semelhante ao GeoJSON.
        hex_list.extend(h3.polyfill(geojson=temp, res=initial_aperture_size))

    # Nome da coluna com o tamanho da abertura
    initial_hex_col = 'hex{}'.format(initial_aperture_size)
    final_hex_col = 'hex{}'.format(final_aperture_size)

    # Criando DataFrame com índices hexagonais
    df_hex = gpd.GeoDataFrame(hex_list, columns=[initial_hex_col])

    # Converter para abertura 8
    df_hex[final_hex_col] = df_hex[initial_hex_col]\
        .apply(lambda x: h3.h3_to_parent(x, 8))

    # Descartando colunas com abertura original
    df_hex.drop(columns=[initial_hex_col], inplace=True)

    # Criando polígonos hexágonos com base em índices hexágonos
    df_hex['geometry'] = df_hex[final_hex_col]\
        .apply(lambda x: Polygon(h3.h3_to_geo_boundary(x, geo_json=True)))

    # Configurando a geometria GeoDataFrame
    df_hex.set_geometry('geometry', crs=area.crs, inplace=True)

    # Queda de hexágonos duplicados causados quando convertemos a abertura
    df_hex.drop_duplicates(inplace=True)

    # Gerando DataFrame com centroides dos hexágonos
    centroides_hex = df_hex['geometry'].centroid
    df_centroides_hex = gpd.GeoDataFrame(df_hex['hex8'],
                                         geometry=centroides_hex,
                                         crs=df_hex.crs)

    return df_centroides_hex, df_hex


def gini(x: np.ndarray) -> float:
    """
    Calcula o índice de Gini para um array de valores.

    O índice de Gini é uma medida de desigualdade de distribuição de valores
    em um conjunto de dados.
    Um valor de Gini igual a 0 indica perfeita igualdade, enquanto um valor
    próximo a 1 indica perfeita desigualdade.

    Args:
        x (np.ndarray): O array de valores para o qual o índice de Gini
        será calculado.

    Returns:
        float: O índice de Gini calculado para o array.
    """
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))


def gini_geral(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o índice de Gini para cada feature em um DataFrame e retorna
    um DataFrame com os resultados.

    Args:
        df (pd.DataFrame): O DataFrame que contém os dados.

    Returns:
        pd.DataFrame: Um DataFrame com as seguintes colunas:
            * "Feature": Nomes das features.
            * "GINI": Valores do índice de Gini para cada feature.

    Observações:
        * A função assume que a coluna `cluster_column` contém
        valores categóricos.
        * A função utiliza a função `gini` para calcular o índice de Gini
        para cada feature.
        * O DataFrame retornado é ordenado pelos valores do índice de Gini.
    """

    gini_list = []
    cols = []

    for col in list(df.columns):
        cols.append(col)
        gini_list.append(gini(df[col].values))

    return pd.DataFrame({"Feature": cols, "GINI": gini_list})\
        .sort_values("GINI")


def gini_cluster(df: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    """
        Calcula o índice de Gini para cada feature em cada cluster e retorna
        um DataFrame com os resultados.

        Args:
            df (pd.DataFrame) : O DataFrame que contém os dados.
            cluster_column (str): O nome da coluna que contém os labels
            dos clusters.

        Returns:
            pd.DataFrame: Um DataFrame com as seguintes colunas:
                * "Feature": Nomes das features.
                * "GINI cluster N": Valores do índice de Gini para cada
                feature em cada cluster (N representa o número do cluster).

        Observações:
            * A função assume que a coluna `cluster_column` contém
            valores categóricos.
            * A função utiliza a função `gini` (que não está definida
            neste código) para calcular o índice de Gini para cada feature.
            * O DataFrame retornado contém uma coluna para cada cluster, com
            o nome "GINI cluster N", onde N é o número do cluster.
            * As colunas do DataFrame são ordenadas primeiro pelas features
            e depois pelos clusters.
    """

    clusters = list(df[cluster_column].unique())
    cols = []

    for col in list(df.drop(columns=[cluster_column])):
        cols.append(col)

    gini_table = pd.DataFrame({"Feature": cols})

    for cluster in clusters:
        df_cluster = df[df[cluster_column] == cluster]\
            .drop(columns=[cluster_column])
        gini_values = []
        for col in list(df_cluster):
            gini_values.append(gini(df_cluster[col].values))
        gini_table['GINI cluster '+str(cluster)] = gini_values
    return gini_table


def evaluate_gini_clusters(df: pd.DataFrame,
                           cluster_name: str,
                           columns: list) -> pd.DataFrame:
    """
        Avalia o índice de Gini para cada feature em cada cluster e retorna
        um DataFrame com estatísticas adicionais.

        Args:
            df (pd.DataFrame): O DataFrame que contém os dados.
            cluster_name (str): O nome da coluna que contém os labels
            dos clusters.
            columns (list): Uma lista de strings representando os nomes
            das features a serem analisadas.

        Returns:
            pd.DataFrame: Um DataFrame com as seguintes colunas:\n
                * "Feature": Nomes das features.
                * "GINI": Valores médios do índice de Gini para cada
                feature por cluster.
                * "Var": Variância das features para cada cluster.
                * "mean": Média geral das features.
                * "std": Desvio padrão geral das features.
                * "CV": Coeficiente de Variação (CV) das features
                (desvio padrão / média) para cada cluster.

        Observações:
            * A função assume que a coluna `cluster_name` contém
            valores categóricos.
            * A função utiliza as funções `gini_geral` e `gini_cluster`
            (que não estão definidas neste código)
            para calcular o índice de Gini para cada feature.
            * O DataFrame retornado contém estatísticas adicionais para
            cada feature em cada cluster,
            incluindo variância, média geral, desvio padrão geral
            e coeficiente de variação.
    """
    list_features_gini = columns.copy()
    list_features_gini.append(cluster_name)
    df_slice = df[list_features_gini]
    cluster_means = df_slice.groupby(cluster_name).mean()
    centroids = pd.DataFrame(round(cluster_means.var(), 4), columns=['Var'])\
        .reset_index()
    mean = cluster_means.mean()
    std = cluster_means.std(ddof=0)
    centroids['mean'] = round(mean, 4).values
    centroids['std'] = round(std, 4).values
    centroids['CV'] = round((std)/(mean), 4).values
    gine_geral_table = round(gini_geral(df[columns]), 3)
    gine_view = round(gini_cluster(df_slice, cluster_name), 3)\
        .merge(gine_geral_table, on='Feature')\
        .merge(centroids, left_on='Feature', right_on='index')
    gine_view.drop(columns=['index'], inplace=True)
    return gine_view
