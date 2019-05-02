# S2RDF

S2RDF (SPARQL on Spark for RDF) is a SPARQL query processor for Hadoop based on Spark SQL. It uses the relational interface of Spark for query execution and comes with a novel partitioning schema for RDF called ExtVP (Extended Vertical Partitioning) that is an extension of the Vertical Partitioning (VP) schema introduced by Abadi et al. ExtVP enables to exclude unnecessary data from query processing by taking into account the possible relations between tables in VP.

http://dbis.informatik.uni-freiburg.de/forschung/projekte/DiPoS/S2RDF.html


### LICENSE
Unless explicitly stated otherwise all files in this repository are licensed under the Apache Software License 2.0

>   Copyright 2017 University of Freiburg
>
>   Licensed under the Apache License, Version 2.0 (the "License");
>   you may not use this file except in compliance with the License.
>   You may obtain a copy of the License at
>
>       http://www.apache.org/licenses/LICENSE-2.0
>
>   Unless required by applicable law or agreed to in writing, software
>   distributed under the License is distributed on an "AS IS" BASIS,
>   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
>   See the License for the specific language governing permissions and
>   limitations under the License.
