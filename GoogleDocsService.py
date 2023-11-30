from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']


class GoogleDocsService:
    def __init__(self):
        self.service = self.create_service()

    def create_service(self):
        """Shows basic usage of the Docs API.
        Prints the title of a sample document.
        """
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    '/home/sergiov/Downloads/client_secret_523128072497-80o6vkcpc0ecgum1fa59lhhb88h6s7ir.apps.googleusercontent.com (1).json',
                    SCOPES)
                creds = flow.run_local_server()
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        service = build('docs', 'v1', credentials=creds)

        return service

    def create_table(self, rows, columns, table_index):
        table_body = {
            "insertTable":
                {
                    "rows": rows,
                    "columns": columns,
                    "location":
                        {
                            "index": table_index
                        }
                }
        }
        return table_body

    def create_document(self, title):
        body = {'title': title}
        doc = self.service.documents().create(body=body).execute()
        title = doc.get('title')
        _id = doc.get('documentId')
        print(f'Created document with title: {title}, id: {_id}')

        return _id

    def insert_table_data(self, rows, columns, text):
        table_data = []
        for i in reversed(range(rows)):
            inp_text = text[i]
            rowIndex = (i + 1) * 5
            for j in reversed(range(columns)):
                index = rowIndex + (j * 2)
                insert_value = {
                    "insertText":
                        {
                            "text": inp_text[j],
                            "location":
                                {
                                    "index": index
                                }
                        }

                }
                table_data.append(insert_value)

        return table_data

    def insert_title(self, file_id, content):
        requests = [
            {
                'insertText': {
                    'text': f'\n{content}\n',
                    'location': {
                        'index': 1,
                    },
                },
            },
            {
                'updateParagraphStyle': {
                    'paragraphStyle': {
                        'namedStyleType': 'HEADING_2',
                    },
                    'range': {
                        'startIndex': 2,
                        'endIndex': len(content) + 2,
                    },
                    'fields': 'namedStyleType',
                },
            },
        ]

        self.service.documents().batchUpdate(documentId=file_id, body={'requests': requests}).execute()

    def insert_text(self, file_id, content):
        requests = [
            {
                'insertText': {
                    'text': f'{content}',
                    'location': {
                        'index': 1,
                    },
                }
            }
        ]

        # Execute the requests
        self.service.documents().batchUpdate(documentId=file_id, body={'requests': requests}).execute()

    def insert_feature_importances_table(self, file_id, importances_1, importances_2):
        requests = []

        requests.append(self.create_table(2, 2, 1))

        text = [['Importances train test', 'Importances internal validation'], [importances_1, importances_2]]
        requests.append(self.insert_table_data(2, 2, text))

        self.service.documents().batchUpdate(documentId=file_id, body={'requests': requests}).execute()
