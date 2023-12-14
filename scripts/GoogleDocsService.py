from __future__ import print_function
import pickle
import os.path

from gdoctableapppy import gdoctableapp
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']


class GoogleDocsService:
    def __init__(self, secret_path, token_path=''):
        """
        Instantiates the Google Docs Service

        Args:
            secret_path (string): path to the API secret.
            token_path (string): if already created, path to the API token, if not, path to the desired destination
                                 of the token.
        """
        self.secret_path = secret_path
        self.token_path = token_path

        self.creds, self.service = self.create_service()

    def create_service(self):
        """
        Creates the API token if it does not exist or retrieves it from the path in order to create an instance of
        the Google Docs Service

        Returns:
            The needed credentials for the API interaction and the service.

        """
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except:
                    flow = InstalledAppFlow.from_client_secrets_file(self.secret_path, SCOPES)
                    creds = flow.run_local_server()
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.secret_path, SCOPES)
                creds = flow.run_local_server()
            # Save the credentials for the next run
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)

        service = build('docs', 'v1', credentials=creds)

        return creds, service

    def create_table(self, file_id, rows, columns, table_data):
        """
        Creates a table in a Google Docs document.

        Args:
            file_id (str): The ID of the Google Docs document.
            rows (int): The number of rows in the table.
            columns (int): The number of columns in the table.
            table_data (list): The data to populate the table.
        """
        request = {
            "oauth2": self.creds,
            "documentId": file_id,
            "rows": rows,
            "columns": columns,
            "createIndex": 1,
            "values": table_data
        }

        gdoctableapp.CreateTable(request)

    def create_document(self, title):
        """
        Creates a new Google Docs document with the specified title.

        Args:
            title (str): The title of the new document.

        Returns:
            str: The ID of the newly created document.
        """
        folder_id = '1X4zrCTu-PwLTBDGhF-UdTClsQxv9svIT'
        body = {
            'title': title
        }
        doc = self.service.documents().create(body=body).execute()

        title = doc.get('title')
        _id = doc.get('documentId')
        print(f'Created document with title: {title}, id: {_id}')

        return _id

    def insert_heading(self, file_id, content, heading_type):
        """
        Inserts a heading with the specified content and heading type into a Google Docs document.

        Args:
            file_id (str): The ID of the Google Docs document.
            content (str): The content of the heading.
            heading_type (str): The type of heading to be applied.
        """
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
                        'namedStyleType': heading_type,
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
        """
        Inserts text into a Google Docs document.

        Args:
            file_id (str): The ID of the Google Docs document.
            content (str): The text to be inserted.
        """
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
