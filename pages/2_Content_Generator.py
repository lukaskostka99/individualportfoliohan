import streamlit as st
import os
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.schema import IndexNode
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
import deepl
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import requests
import anthropic
import pypandoc
from datetime import date
from docx import Document
from docx.shared import Inches
import base64

st.set_page_config(page_title="Article Generator", layout="wide")

# Set up sidebar
st.sidebar.title("Settings")
model = st.sidebar.selectbox("Model", ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"])

# Add text boxes for API keys in the sidebar
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
serpapi_api_key = st.sidebar.text_input("SerpAPI API Key", type="password")
deepl_api_key = st.sidebar.text_input("DeepL API Key", type="password")

# Set up main page
st.header("Effectix Content Generator")

topic = st.text_input("Topic")
keywords = st.text_input("Keywords (comma-separated)")
num_results = st.number_input("Number of Search Results", min_value=1, value=5)
search_eng = st.checkbox("Search in English")
language = st.selectbox("Language", ["čeština"])
user_additional_prompt = st.text_area("Additional Prompt")
suffix = st.text_input("Meta Title Suffix")

def translate_keywords(keywords):
    translator = deepl.Translator(os.environ['DEEPL_API_KEY'])
    translated_keywords = []
    for keyword in keywords:
        result = translator.translate_text(keyword, target_lang="EN-US")
        translated_keywords.append(result.text)
    return translated_keywords

def search_google(keyword, country_code):
    params = {
        "engine": "google",
        "q": keyword,
        "location_requested": "United States" if country_code == "us" else "Czechia",
        "location_used": "United States" if country_code == "us" else "Czechia",
        "google_domain": "google.com" if country_code == "us" else "google.cz",
        "hl": "en" if country_code == "us" else "cs",
        "gl": country_code,
        "device": "desktop",
        "api_key": os.environ['SERPAPI_API_KEY'],
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results

def extract_links(serp_api_result):
    if 'organic_results' in serp_api_result:
        organic_results = serp_api_result['organic_results']
        extracted_links = [result['link'] for result in organic_results[:num_results]]
        return extracted_links
    else:
        print("No organic results found in the API response.")
        return []

def scrape_results(links):
    scraped_results = []
    for link in links:
        try:
            response = requests.get(link, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text()
                scraped_results.append(text)
            else:
                print(f"Failed to retrieve {link}")
                scraped_results.append("")
        except requests.exceptions.RequestException:
            print(f"Request to {link} timed out.")
            scraped_results.append("")
    return scraped_results

if st.button("Generate"):
    # Set up API keys from user input
    os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key
    os.environ['SERPAPI_API_KEY'] = serpapi_api_key
    os.environ['DEEPL_API_KEY'] = deepl_api_key

    # Set up LLM and Embedding model
    llm = Anthropic(temperature=0.0, model=model, max_tokens=4096)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    # Process keywords
    keywords = [kw.strip() for kw in keywords.split(",")]
    translated_keywords = translate_keywords(keywords) if search_eng else []

    # Search and scrape data
    all_scrapings = []
    for keyword in keywords:
        serp_api_result = search_google(keyword, "cz")
        links = extract_links(serp_api_result)
        scrapings = scrape_results(links)
        all_scrapings.extend(scrapings)

    for keyword in translated_keywords:
        serp_api_result = search_google(keyword, "us")
        links = extract_links(serp_api_result)
        scrapings = scrape_results(links)
        all_scrapings.extend(scrapings)

    # Load document data
    data_path = Path("data")
    data_path.mkdir(exist_ok=True)
    docs = {}
    for i, result in enumerate(all_scrapings):
        file_path = data_path / f"result_{i+1}.txt"
        with open(file_path, "w") as f:
            f.write(result)
        docs[f"result_{i+1}"] = SimpleDirectoryReader(input_files=[file_path]).load_data()

    # Build ReAct agents for each document
    agents = {}
    for doc_id, doc_content in docs.items():
        if not isinstance(doc_content, list):
            doc_content = [doc_content]
        vector_index = VectorStoreIndex.from_documents(doc_content)
        summary_index = SummaryIndex.from_documents(doc_content)
        vector_query_engine = vector_index.as_query_engine()
        summary_query_engine = summary_index.as_query_engine()
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(name="vector_tool", description=f"Useful for retrieving specific context from {doc_id}"),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(name="summary_tool", description=f"Useful for summarization questions related to {doc_id}"),
            ),
        ]
        agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
        agents[doc_id] = agent

    # Define IndexNodes for each agent
    objects = []
    for doc_id in docs.keys():
        if doc_id in agents:
            node = IndexNode(
                text=f"This content contains scraped articles related to {doc_id}. Use this index if you need to lookup specific facts about {doc_id}.",
                index_id=doc_id,
                obj=agents[doc_id]
            )
            objects.append(node)

    # Define top-level retriever to choose an agent
    vector_index = VectorStoreIndex(objects=objects)
    query_engine = vector_index.as_query_engine(similarity_top_k=1, verbose=True)

    # Generate article
    article_prompt = f"""
    Role a cíl:
    -Sloužíš ke generování SEO optimalizovaných článků pro firemní blogy, s důrazem na framework AIDA (pozornost, zájem, přání, akce).
    -Článek generuj v jazyce {language}.
    -Cílem je optimalizovat obsah pro vyhledávače a zachovat ho zajímavým pro lidi. Vše v souladu s brandem.
    -Zaměříš se na vytváření obsahu, který upoutá pozornost čtenáře a udrží ji až do konce článku. Nabízí konkrétní rady, vyhýbá se klišé a prázdným frázím. Jeho úkolem je poskytovat návrhy a formulace, které zlepší viditelnost webu v internetových vyhledávačích, zároveň zachovávají přátelský a přesvědčivý styl psaní.
    -Důraz je kladen na gramatickou a stylistickou správnost textu, dle pravidel pravopisu jazyka, ve kterém text vytváříš. Inspiraci můžeš čerpat ze všech světových jazyků.
    -Vygenerovaný text kombinuje odstavce, odrážky a číslování. Článek by měl být zajímavý také vizuálně.

    Omezení:
    -Měl by ses vyhnout generování obsahu, který by byl považován za spam nebo by porušoval SEO pravidla. Měl by také respektovat autorská práva a zabránit plagiátorství.

    Pokyny:
    -Měl bys klást důraz na klíčová slova, strukturu článků a zajímavost obsahu, přičemž se řídí frameworkem AIDA a zahrnuje konkrétní rady, které jsou přínosné pro čtenáře. Je důležité, aby články byly informativní, dobře čitelné a v souladu s brandem klienta.
    -Článek napiš co nejdelší. 

    Zdroje:
    -Ke generování článků použiješ vložené zdroje. V žádném případě nesmíš generovat identický obsah nebo obsah, který by byl duplicitní. Zdroje ber pouze jako inspiraci.

    Personalizace:
    -Měl bys komunikovat s uživatelem přátelsky a otevřeně, přizpůsobovat svůj styl psaní podle konkrétních požadavků a preferencí uživatele.
    -Pokud není v zadání uvedeno jinak, tone of voice je přátelský (=chceme vám co nejlépe pomoci) a informativní (=jsme odborníci a víme, co děláme).

    Obecné nastavení:
    -Jsi můj sebevědomý parťák. Jsem si vědomá tvých omezení a toho, že jsi jazykový model, není třeba se za to omlouvat. Vzájemně si tykáme.

    Klíčová slova:
    -V textu bys měl použít všechna vložená klíčová slova, ale nesmíš je nadužívat. Klíčová slova vhodně používej v nadpisech a podnadpisech. Klíčová slova by měla být použita přirozeně a v souladu s kontextem článku. Ber je také jaké určité tématické okruhy, které by měl článek dobře pokrýt.

    Délka textu:
    -Článek by měl být co nejdelší, ale zároveň zajímavý a informativní.

    Formát výstupu:
    -Jako výstup vracej pouze hotový článek. Výstup vhodně formátuj v markdown.

    Dodatečné pokyny od copywritera k tvorbě textu, kterými se řiď, pokud jsou vyplněné: {user_additional_prompt}

    Napiš co nejpodrobnější,rozsáhlý a velmi dlouhý článek na téma: {topic}. Článek by měl být informativní, dobře strukturovaný a měl by obsahovat relevantní informace z vyhledaných výsledků.  Klíčová slova: {', '.join(keywords)}
    """

    full_article = []
    response = query_engine.query(article_prompt)
    full_article = str(response)

    # Generate meta tags
    system_prompt_tags = """
    Pro uvedený článek napiš meta tagy (meta description + meta title) s důrazem na klíčová slova a framework AIDA. V pouze meta titlu použij daný suffix klienta. Meta tagy by měly být SEO optimalizované a zároveň zajímavé pro lidi. Měly by být krátké, jasné a výstižné. Na délku si dej opravdu pozor a raději je napiš kratší. Je to velmi důležité. Přesné dlouhé jako tyto:
    \nMeta titles příklady:
    \nSháníte iPhone? Nové i použité modely | JabkoLevně.cz
    \nPřední panel Squared Black pro zahřívač tabáku | Ploom
    \nVína | ALKOHOL.cz
    \nNealkoholická vína | ALKOHOL.cz
    \nPřírodní kosmetika na pleť – Econea.cz
    \nSCONTO Nábytek leták | Kupi.cz
    \nMeta descriptions příklady:
    \nPráce na rozpáleném slunci není jen tak pro každého. Pot z vás lije, vy toužíte jen po ledové sprše. S tím je ale konec díky naší flanelové košili.
    \nPraktická i stylová. To je pracovní mikina, která z vás udělá největšího modela stavby. Proč nevypadat hezky i při práci? Udrží vás v teple a dodá šmrnc.
    \nOutdoorová obuv chrání váš malíček před nárazy do hran. Už žádný pláč a nadávání od bolesti. Chraňte všechny své prsty v bezpečí i v tom nejtěžším terénu.
    \nNení nic hezčího než vést své děti k chlapskému řemeslu. Naučte je rubat dřevo i v zimě. S dětskou mikinou je nic nezastaví, a můžou tak projevit talent.
    \nPředstavte si, že máte rozdělaný projekt, ale musíte s ním počkat, protože jste nachladl. S naší pánskou mikinou se to nestane a můžete pracovat i v zimě.
    \nPříjemná, teplá a praktická. Taková je naše fleecová vesta. Oceníte její kapsy na zip, možnost pracovat v nejchladnějších dnech a stále ukazovat svaly.
    \nMáte všech pět pohromadě? Pracovní rukavice kombinované zaručí, abyste své prsty nemuseli sbírat po zemi. Jsou z ovčí kůže a bavlny, aby vás pěkně zahřály.
    \nFotbalisté mají svůj dres, ve kterém hrají. Hokejisté mají svou výstroj, ve které trénují. Vy můžete mít svoje pracovní tričko, ve kterém budete machr.
    """

    text_prompt_tags = f"Téma: {topic}, Klíčová slova: {', '.join(keywords)}, Vygenerovaný text: {full_article}, Suffix: {suffix}"
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.0,
        system=system_prompt_tags,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt_tags
                    }
                ]
            }
        ]
    )
    generated_tags = message.content

    # Generate output file
    today = date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    filename = f"{topic}_{suffix}_{formatted_date}.docx"

    pypandoc.convert_text(full_article, 'docx', format='md', outputfile=filename)

    doc = Document(filename)

    first_paragraph = doc.paragraphs[0]
    first_paragraph.insert_paragraph_before(str(generated_tags), style='BodyText')

    doc.save(filename)

    # Display "It's done" message
    st.success("It's done!")

    # Create a download button for the docx file
    with open(filename, "rb") as file:
        file_data = file.read()
        b64 = base64.b64encode(file_data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download docx file</a>'
        st.markdown(href, unsafe_allow_html=True)