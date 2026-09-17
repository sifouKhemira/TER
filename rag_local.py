from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# pour l'instant je fais que du BM25 (recherche par mots-clés classique) : le modèle
# d'embeddings local que j'avais prévu est tout petit et capte pas bien le vocabulaire
# technique du CCTP, du coup en attendant de régler ça je pars sur BM25 tout seul pour
# avoir déjà un premier test qui marche
from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever # Désactivé temporairement suite à l'erreur d'import

print("1. Ingestion du CCTP...")
loader = PDFPlumberLoader("19082-COMP-CCTP-001.pdf")  # le PDF à traiter, en dur pour le moment
docs = loader.load()  # charge tout le texte du PDF en mémoire (une entrée par page je crois)

print("2. Découpage en blocs (Chunking)...")
# 1000 caractères par bloc avec 150 de chevauchement : valeurs prises un peu au pif
# pour commencer, à ajuster si les réponses sont pas top
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = text_splitter.split_documents(docs)  # le doc entier devient une liste de petits chunks

print("3. Création des moteurs de recherche (BM25 seul pour le test)...")
# B. Moteur par Mots-Clés (BM25)
retriever = BM25Retriever.from_documents(splits)
retriever.k = 6  # nombre de blocs remontés par recherche, à tester avec d'autres valeurs

print("4. Configuration du LLM Mistral...")
llm = OllamaLLM(model="mistral")  # Mistral tourne en local via Ollama

# c'est LE prompt important : sans cette consigne le modèle a tendance à
# "compléter" avec ses propres connaissances au lieu de rester sur le PDF,
# donc je force bien à dire "je sais pas" plutôt que d'inventer un truc faux
system_prompt = (
    "Tu es un ingénieur expert. "
    "Réponds à la question de l'utilisateur en te basant STRICTEMENT sur le contexte fourni. "
    "Si la réponse n'y figure pas, dis uniquement 'Je ne sais pas', sans faire de suppositions ni de déductions..\n\n"
    "Contexte : {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
# assemble le message final envoyé au LLM (consigne système + question de l'utilisateur)


# Fonction pour formater les textes trouvés
def format_docs(docs):
    # les blocs récupérés arrivent comme une liste d'objets Document,
    # je les recolle juste en un seul gros texte avec des sauts de
    # ligne entre chaque, plus simple à donner au prompt
    return "\n\n".join(doc.page_content for doc in docs)


# la chaine RAG complète : recherche -> mise en forme du contexte -> prompt -> LLM -> texte propre
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt  # injecte contexte + question dans le template
    | llm     # envoie le tout à Mistral
    | StrOutputParser()  # récupère juste le texte de la réponse (pas tout l'objet retourné par le LLM)
)

print("\n--- TEST DU RAG ---")
question = "Quelles sont les caractéristiques exigées pour le complexe d'étanchéité VULCASTEEL ROOF ?"
question2 = "Quelle est la surface géométrique minimale pour l'exutoire de fumée du local Etuves ?"
question3 = "Quelles sont les obligations de l'entreprise en matière de gestion des déchets et d'EPI ?"
print(f"Question : {question3}\n")

# Lancement de la recherche et de la génération
print("\n--- DEBUG : CE QUE LE MODELE LIT ---")
# petit affichage de debug pour voir concrètement quels blocs BM25 a retrouvé
# avant de les envoyer au LLM -> hyper utile pour comprendre pourquoi une
# réponse est fausse ou incomplète (souvent c'est juste que le bon passage
# n'a pas été retrouvé du tout)
docs_trouves = retriever.invoke(question3)
for i, doc in enumerate(docs_trouves):
    print(f"Bloc {i+1} : {doc.page_content[:200]}...\n")
response = rag_chain.invoke(question3)  # là on relance tout le pipeline en entier, pas juste la recherche
print("Réponse du modèle :")
print(response)