import sqlite3
from pathlib import Path
from typing import Sequence, Any

__all__ = ["SQLConnector"]


class SQLConnector:
    """
    Class enabling connection with the sql database based on sqlite3.

    Args:
        database_path(str | Path): path to the database.
            If the path does not exist correct db will be created.
    """

    def __init__(self, database_path: str | Path):
        self._database_path = str(database_path)

    def create_table(self, table_name: str, fields: Sequence[str]) -> None:
        """
        Create table in the database with specified name.

        Args:
            table_name(str): name of the table to be created in the database.
            fields(Sequence[str]): names of the fields in the table.

        Returns:
            None
        """
        # let first element be the primary key by default
        connection = sqlite3.connect(str(self._database_path))
        cursor = connection.cursor()
        fields = list(fields)
        fields[0] = f"{fields[0]} PRIMARY KEY"
        query = f"CREATE TABLE IF NOT EXISTS {table_name}({', '.join(fields)})"
        cursor.execute(query)
        connection.close()

    def remove_table(self, table_name: str) -> None:
        """
        Remove table from the database.

        Args:
            table_name(str): table to be removed from the database.

        Returns:
            None
        """
        connection = sqlite3.connect(str(self._database_path))
        cursor = connection.cursor()
        query = "DROP TABLE %s" % table_name
        cursor.execute(query)
        connection.close()

    def get_existing_tables(self) -> tuple[str, ...]:
        """
        Get names of existing tables in sqlite database.

        Returns:
            tuple[str, ...]: existing tables.
                Example: ('users', 'products')
        """
        connection = sqlite3.connect(str(self._database_path))
        cursor = connection.cursor()
        query = "SELECT name FROM sqlite_master"
        res = cursor.execute(query).fetchone()
        connection.close()
        return res

    def add_to_table(self, table_name: str, records: list[tuple]) -> None:
        """
        Add records to the chosen table.

        Args:
            table_name(str): name of the table to which records will be added.
            records(list[tuple]): records to be added.
                Example: [(field_1_value, field_2_value, ...), ...]

        Returns:
            None
        """
        connection = sqlite3.connect(str(self._database_path))
        cursor = connection.cursor()
        query = f"INSERT INTO {table_name} VALUES({', '.join(['?']*len(records[0]))})"

        cursor.executemany(query, records)
        connection.commit()

        connection.close()

    def get_record_from_table(self, table_name: str, key_name: str, key_val: Any) -> dict[str, Any]:
        """
        Get record from the chosen table with given primary key.
        if the given key won't be primary returned result will be
        the first occurrence satisfying requirement.

        Args:
            table_name(str): name of the table to get records from.
            key_name(str): name of the key in the table (e.g. id or username).
            key_val(Any): what value we are looking for.

        Returns:
            dict[str, Any]: output data with chosen record as a dict.
        """
        connection = sqlite3.connect(str(self._database_path))
        cursor = connection.cursor()

        query = f"SELECT * FROM {table_name} WHERE {key_name} = ?"
        res = cursor.execute(query, [key_val])
        tuple_res = res.fetchone()
        keys = [desc[0] for desc in res.description]
        res_dict = {k: v for k, v in zip(keys, tuple_res)} if tuple_res else {}
        connection.close()
        return res_dict

    def get_table(self, table_name: str) -> dict[str, Any]:
        """
        Get full table as dict.

        Args:
            table_name(str): name of the table to be returned.

        Returns:
            dict[str, list[Any]]: table transformed to a dictionary.
        """
        connection = sqlite3.connect(str(self._database_path))
        cursor = connection.cursor()

        query = f"SELECT * from {table_name}"
        res = cursor.execute(query)
        many = res.fetchall()
        keys = [desc[0] for desc in res.description]
        result_dict = {key: [] for key in keys}

        for record in many:
            for key, value in zip(keys, record):
                result_dict[key].append(value)
        connection.close()

        return result_dict

    def update_record_in_table(
        self, table_name: str, primary_key_name: str, primary_key_val: Any, key_name: str, new_value: Any
    ) -> None:
        """
        Update a specific value in the chosen table for a given primary key.

        Args:
            table_name(str): name of the table to update records in.
            primary_key_name(str): name of the primary key column in the table (e.g. id).
            primary_key_val(Any): the current value of the primary key to be updated.
            key_name(str): the column to be updated.
            new_value(Any): the new value to be set.

        Returns:
            None
        """
        connection = sqlite3.connect(str(self._database_path))
        cursor = connection.cursor()

        query = f"UPDATE {table_name} SET {key_name} = ? WHERE {primary_key_name} = ?"
        cursor.execute(query, (new_value, primary_key_val))
        connection.commit()
        connection.close()

    def remove_from_table(self, table_name: str, column_name: str, value: any) -> None:
        """
        Remove records from the chosen table based on the given column and value.

        Args:
            table_name (str): name of the table from which records will be removed.
            column_name (str): name of the column to use in the condition.
            value (any): value to match for the condition.

        Returns:
            None
        """
        connection = sqlite3.connect(str(self._database_path))
        cursor = connection.cursor()

        query = f"DELETE FROM {table_name} WHERE {column_name} = ?"

        cursor.execute(query, (value,))
        connection.commit()
        connection.close()
