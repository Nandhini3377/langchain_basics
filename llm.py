from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser,StrOutputParser,JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

#initiate model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# #promt template first type
# prompt=ChatPromptTemplate.from_template("Tell me a joke {val}")
# chain=prompt|llm
# response=chain.invoke({"val":"cat"})
# print(response)


# #promt template sec type more specific way
# prompt=ChatPromptTemplate.from_messages([
#     ("system","You are Phd Holded English Trainer.Give me some vocalubary under category in comma separated format"),
#     ("human","{category}")
# ])

# parser=CommaSeparatedListOutputParser()
# chain=prompt|llm|parser
# response=chain.invoke({"category":"office"})
# print(response)


class Product(BaseModel):
    productname: str = Field(description="The name of the product")
    price: int = Field(description="The price of the product")

parser = JsonOutputParser(pydantic_object=Product)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract product details as a JSON object with fields: productname and price."),
    ("human", "{category}")
])
chain = prompt | llm | parser


response = chain.invoke({
    "category": "The sizzler brownie is 300 rupees in Indian amount.",
    "format": parser.get_format_instructions()
})
print(response)



# response=llm.stream("meaning of exhaust?");
# for chunk in response:
#     print(chunk.content,end='',flush=True)