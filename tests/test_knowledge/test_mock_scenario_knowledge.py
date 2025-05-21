# # test_complex_scenario_knowledge.py

# import os
# import json
# import pytest
# import datetime
# import time
# from typing import List, Dict, Any, Optional
# import unify

# from unity.knowledge_manager.knowledge_manager import KnowledgeManager


# class TestComplexKnowledgeScenario:
#     """Complex real-world test scenarios for KnowledgeManager with live LLM integration."""

#     @classmethod
#     def setup_class(cls):
#         """Initialize test resources once at class level."""
#         cls.knowledge_manager = KnowledgeManager()

#         # Seed initial knowledge base with complex, interconnected data
#         cls._seed_knowledge_base()

#     @classmethod
#     def teardown_class(cls):
#         """Clean up test resources."""
#         # Clean up created tables
#         tables = cls.knowledge_manager._list_tables()
#         for table in tables:
#             cls.knowledge_manager._delete_table(table)

#     @classmethod
#     def _seed_knowledge_base(cls):
#         """Create a comprehensive knowledge base with multiple interconnected tables."""

#         # 1. Create Clients table
#         cls.knowledge_manager.store(
#             "Create a table called 'Clients' with columns for client_id (integer), "
#             "name (string), industry (string), size (string), relationship_start_date (date), "
#             "primary_contact_name (string), primary_contact_email (string), and status (string)."
#         )

#         # Add client data
#         clients_data = [
#             {
#                 "client_id": 1,
#                 "name": "TechNova Systems",
#                 "industry": "Technology",
#                 "size": "Enterprise",
#                 "relationship_start_date": "2020-03-15",
#                 "primary_contact_name": "Sarah Johnson",
#                 "primary_contact_email": "sarah.johnson@technova.com",
#                 "status": "Active",
#             },
#             {
#                 "client_id": 2,
#                 "name": "HealthFirst Medical Group",
#                 "industry": "Healthcare",
#                 "size": "Mid-market",
#                 "relationship_start_date": "2021-07-22",
#                 "primary_contact_name": "Michael Chang",
#                 "primary_contact_email": "michael.chang@healthfirst.org",
#                 "status": "Active",
#             },
#             {
#                 "client_id": 3,
#                 "name": "Global Financial Services",
#                 "industry": "Finance",
#                 "size": "Enterprise",
#                 "relationship_start_date": "2019-11-05",
#                 "primary_contact_name": "Lisa Williams",
#                 "primary_contact_email": "lwilliams@globalfin.com",
#                 "status": "Active",
#             },
#             {
#                 "client_id": 4,
#                 "name": "Eco Solutions",
#                 "industry": "Environmental",
#                 "size": "Small Business",
#                 "relationship_start_date": "2022-01-10",
#                 "primary_contact_name": "James Rodriguez",
#                 "primary_contact_email": "james.rodriguez@ecosolutions.net",
#                 "status": "Active",
#             },
#             {
#                 "client_id": 5,
#                 "name": "Retail Innovations",
#                 "industry": "Retail",
#                 "size": "Mid-market",
#                 "relationship_start_date": "2021-04-30",
#                 "primary_contact_name": "Emma Davis",
#                 "primary_contact_email": "edavis@retailinnovations.com",
#                 "status": "Active",
#             },
#             {
#                 "client_id": 6,
#                 "name": "Construction Partners",
#                 "industry": "Construction",
#                 "size": "Small Business",
#                 "relationship_start_date": "2022-09-18",
#                 "primary_contact_name": "David Wilson",
#                 "primary_contact_email": "dwilson@constructpartners.com",
#                 "status": "Prospective",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following client data to the Clients table: {json.dumps(clients_data)}"
#         )

#         # 2. Create Projects table
#         cls.knowledge_manager.store(
#             "Create a table called 'Projects' with columns for project_id (integer), "
#             "client_id (integer), name (string), description (text), start_date (date), "
#             "end_date (date), budget (float), status (string), project_manager (string), "
#             "priority (string), and domain (string)."
#         )

#         # Add project data
#         projects_data = [
#             {
#                 "project_id": 101,
#                 "client_id": 1,
#                 "name": "Cloud Migration Initiative",
#                 "description": "Migrating TechNova's legacy infrastructure to a cloud-based microservices architecture with enhanced security protocols",
#                 "start_date": "2023-02-15",
#                 "end_date": "2023-12-30",
#                 "budget": 450000.00,
#                 "status": "In Progress",
#                 "project_manager": "Jennifer Adams",
#                 "priority": "High",
#                 "domain": "Cloud Infrastructure",
#             },
#             {
#                 "project_id": 102,
#                 "client_id": 2,
#                 "name": "Patient Management System",
#                 "description": "Developing a HIPAA-compliant electronic health record system with real-time analytics and interoperability features",
#                 "start_date": "2023-07-01",
#                 "end_date": "2024-03-31",
#                 "budget": 380000.00,
#                 "status": "In Progress",
#                 "project_manager": "Robert Chen",
#                 "priority": "Critical",
#                 "domain": "Healthcare Software",
#             },
#             {
#                 "project_id": 103,
#                 "client_id": 3,
#                 "name": "Fraud Detection Platform",
#                 "description": "Implementing advanced AI-based fraud detection system for financial transactions with regulatory compliance features",
#                 "start_date": "2023-05-10",
#                 "end_date": "2023-11-15",
#                 "budget": 275000.00,
#                 "status": "In Progress",
#                 "project_manager": "Samantha Brooks",
#                 "priority": "Critical",
#                 "domain": "Financial Security",
#             },
#             {
#                 "project_id": 104,
#                 "client_id": 5,
#                 "name": "Omnichannel Retail Platform",
#                 "description": "Creating an integrated e-commerce and in-store retail management system with inventory optimization",
#                 "start_date": "2023-09-01",
#                 "end_date": "2024-06-30",
#                 "budget": 520000.00,
#                 "status": "Planning",
#                 "project_manager": "Thomas Reed",
#                 "priority": "High",
#                 "domain": "Retail Technology",
#             },
#             {
#                 "project_id": 105,
#                 "client_id": 4,
#                 "name": "Sustainability Tracking System",
#                 "description": "Developing a carbon footprint monitoring and reporting system with regulatory compliance dashboards",
#                 "start_date": "2023-04-15",
#                 "end_date": "2023-11-30",
#                 "budget": 195000.00,
#                 "status": "In Progress",
#                 "project_manager": "Alicia Martinez",
#                 "priority": "Medium",
#                 "domain": "Environmental Monitoring",
#             },
#             {
#                 "project_id": 106,
#                 "client_id": 1,
#                 "name": "DevOps Transformation",
#                 "description": "Implementing CI/CD pipelines and containerization for TechNova's development workflows",
#                 "start_date": "2022-10-01",
#                 "end_date": "2023-09-30",
#                 "budget": 325000.00,
#                 "status": "Completed",
#                 "project_manager": "Jennifer Adams",
#                 "priority": "High",
#                 "domain": "DevOps",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following project data to the Projects table: {json.dumps(projects_data)}"
#         )

#         # 3. Create ProjectPhases table
#         cls.knowledge_manager.store(
#             "Create a table called 'ProjectPhases' with columns for phase_id (integer), "
#             "project_id (integer), phase_name (string), start_date (date), end_date (date), "
#             "deliverables (string), status (string), and resources_allocated (float)."
#         )

#         # Add project phases data
#         project_phases_data = [
#             {
#                 "phase_id": 1001,
#                 "project_id": 101,
#                 "phase_name": "Discovery and Assessment",
#                 "start_date": "2023-02-15",
#                 "end_date": "2023-03-31",
#                 "deliverables": "Infrastructure assessment report, migration roadmap",
#                 "status": "Completed",
#                 "resources_allocated": 75000.00,
#             },
#             {
#                 "phase_id": 1002,
#                 "project_id": 101,
#                 "phase_name": "Architecture Design",
#                 "start_date": "2023-04-01",
#                 "end_date": "2023-05-15",
#                 "deliverables": "Cloud architecture blueprints, security protocols",
#                 "status": "Completed",
#                 "resources_allocated": 95000.00,
#             },
#             {
#                 "phase_id": 1003,
#                 "project_id": 101,
#                 "phase_name": "Migration Execution",
#                 "start_date": "2023-05-16",
#                 "end_date": "2023-10-31",
#                 "deliverables": "Migrated applications, test reports",
#                 "status": "In Progress",
#                 "resources_allocated": 220000.00,
#             },
#             {
#                 "phase_id": 1004,
#                 "project_id": 101,
#                 "phase_name": "Optimization and Handover",
#                 "start_date": "2023-11-01",
#                 "end_date": "2023-12-30",
#                 "deliverables": "Performance tuning report, documentation, training",
#                 "status": "Not Started",
#                 "resources_allocated": 60000.00,
#             },
#             {
#                 "phase_id": 1005,
#                 "project_id": 102,
#                 "phase_name": "Requirements Gathering",
#                 "start_date": "2023-07-01",
#                 "end_date": "2023-08-15",
#                 "deliverables": "Detailed requirements document, compliance checklist",
#                 "status": "Completed",
#                 "resources_allocated": 65000.00,
#             },
#             {
#                 "phase_id": 1006,
#                 "project_id": 102,
#                 "phase_name": "System Design",
#                 "start_date": "2023-08-16",
#                 "end_date": "2023-10-15",
#                 "deliverables": "System architecture, database schema, UI/UX designs",
#                 "status": "Completed",
#                 "resources_allocated": 85000.00,
#             },
#             {
#                 "phase_id": 1007,
#                 "project_id": 102,
#                 "phase_name": "Development",
#                 "start_date": "2023-10-16",
#                 "end_date": "2024-01-31",
#                 "deliverables": "Core modules, integration APIs, testing reports",
#                 "status": "In Progress",
#                 "resources_allocated": 150000.00,
#             },
#             {
#                 "phase_id": 1008,
#                 "project_id": 102,
#                 "phase_name": "Deployment and Training",
#                 "start_date": "2024-02-01",
#                 "end_date": "2024-03-31",
#                 "deliverables": "Installed system, training materials, support documentation",
#                 "status": "Not Started",
#                 "resources_allocated": 80000.00,
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following phase data to the ProjectPhases table: {json.dumps(project_phases_data)}"
#         )

#         # 4. Create TeamMembers table
#         cls.knowledge_manager.store(
#             "Create a table called 'TeamMembers' with columns for member_id (integer), "
#             "name (string), role (string), expertise (string), joined_date (date), "
#             "utilization_percentage (float), and current_projects (string)."
#         )

#         # Add team members data
#         team_members_data = [
#             {
#                 "member_id": 1,
#                 "name": "Jennifer Adams",
#                 "role": "Senior Project Manager",
#                 "expertise": "Cloud Migration, DevOps Transformation",
#                 "joined_date": "2018-05-10",
#                 "utilization_percentage": 90.0,
#                 "current_projects": "Cloud Migration Initiative, DevOps Transformation",
#             },
#             {
#                 "member_id": 2,
#                 "name": "Robert Chen",
#                 "role": "Project Manager",
#                 "expertise": "Healthcare Systems, HIPAA Compliance",
#                 "joined_date": "2019-03-22",
#                 "utilization_percentage": 85.0,
#                 "current_projects": "Patient Management System",
#             },
#             {
#                 "member_id": 3,
#                 "name": "Samantha Brooks",
#                 "role": "Senior Project Manager",
#                 "expertise": "Financial Systems, Security",
#                 "joined_date": "2017-11-15",
#                 "utilization_percentage": 95.0,
#                 "current_projects": "Fraud Detection Platform",
#             },
#             {
#                 "member_id": 4,
#                 "name": "Thomas Reed",
#                 "role": "Project Manager",
#                 "expertise": "Retail Systems, E-commerce",
#                 "joined_date": "2020-01-15",
#                 "utilization_percentage": 75.0,
#                 "current_projects": "Omnichannel Retail Platform",
#             },
#             {
#                 "member_id": 5,
#                 "name": "Alicia Martinez",
#                 "role": "Project Manager",
#                 "expertise": "Environmental Systems, Compliance Reporting",
#                 "joined_date": "2021-02-28",
#                 "utilization_percentage": 80.0,
#                 "current_projects": "Sustainability Tracking System",
#             },
#             {
#                 "member_id": 6,
#                 "name": "David Zhang",
#                 "role": "Senior Developer",
#                 "expertise": "Cloud Architecture, Kubernetes, AWS",
#                 "joined_date": "2019-07-15",
#                 "utilization_percentage": 100.0,
#                 "current_projects": "Cloud Migration Initiative",
#             },
#             {
#                 "member_id": 7,
#                 "name": "Emily Johnson",
#                 "role": "UX/UI Designer",
#                 "expertise": "Healthcare UX, Accessibility Design",
#                 "joined_date": "2020-05-20",
#                 "utilization_percentage": 90.0,
#                 "current_projects": "Patient Management System",
#             },
#             {
#                 "member_id": 8,
#                 "name": "Michael Patel",
#                 "role": "Data Scientist",
#                 "expertise": "Machine Learning, Fraud Detection",
#                 "joined_date": "2019-11-10",
#                 "utilization_percentage": 85.0,
#                 "current_projects": "Fraud Detection Platform",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following team member data to the TeamMembers table: {json.dumps(team_members_data)}"
#         )

#         # 5. Create ClientRequirements table with detailed specifications
#         cls.knowledge_manager.store(
#             "Create a table called 'ClientRequirements' with columns for requirement_id (integer), "
#             "project_id (integer), title (string), description (text), priority (string), "
#             "status (string), requested_by (string), and compliance_related (boolean)."
#         )

#         # Add client requirements data
#         client_requirements_data = [
#             {
#                 "requirement_id": 1,
#                 "project_id": 101,
#                 "title": "Zero Downtime Migration",
#                 "description": "The migration must be performed with zero downtime for critical systems. Maintenance windows can be scheduled for non-critical systems with advance notice.",
#                 "priority": "Critical",
#                 "status": "In Progress",
#                 "requested_by": "Sarah Johnson",
#                 "compliance_related": False,
#             },
#             {
#                 "requirement_id": 2,
#                 "project_id": 101,
#                 "title": "Enhanced Security Protocols",
#                 "description": "All migrated systems must implement the latest security protocols including encryption at rest and in transit, role-based access control, and multi-factor authentication.",
#                 "priority": "High",
#                 "status": "In Progress",
#                 "requested_by": "Sarah Johnson",
#                 "compliance_related": True,
#             },
#             {
#                 "requirement_id": 3,
#                 "project_id": 102,
#                 "title": "HIPAA Compliance",
#                 "description": "The patient management system must be fully HIPAA compliant with appropriate access controls, audit logging, and data encryption.",
#                 "priority": "Critical",
#                 "status": "In Progress",
#                 "requested_by": "Michael Chang",
#                 "compliance_related": True,
#             },
#             {
#                 "requirement_id": 4,
#                 "project_id": 102,
#                 "title": "Interoperability Standards",
#                 "description": "The system must support HL7 FHIR standards for interoperability with other healthcare systems and devices.",
#                 "priority": "High",
#                 "status": "Planned",
#                 "requested_by": "Michael Chang",
#                 "compliance_related": True,
#             },
#             {
#                 "requirement_id": 5,
#                 "project_id": 103,
#                 "title": "Real-time Fraud Detection",
#                 "description": "The system must be able to detect fraudulent transactions in real-time with a response time of under 500ms and a false positive rate below 0.1%.",
#                 "priority": "Critical",
#                 "status": "In Progress",
#                 "requested_by": "Lisa Williams",
#                 "compliance_related": False,
#             },
#             {
#                 "requirement_id": 6,
#                 "project_id": 103,
#                 "title": "Regulatory Reporting",
#                 "description": "The system must automatically generate reports for regulatory compliance with SEC, FINRA, and other relevant financial authorities.",
#                 "priority": "High",
#                 "status": "Planned",
#                 "requested_by": "Lisa Williams",
#                 "compliance_related": True,
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following requirement data to the ClientRequirements table: {json.dumps(client_requirements_data)}"
#         )

#         # 6. Create RiskRegistry table to track project risks
#         cls.knowledge_manager.store(
#             "Create a table called 'RiskRegistry' with columns for risk_id (integer), "
#             "project_id (integer), description (text), probability (string), impact (string), "
#             "mitigation_strategy (text), owner (string), and status (string)."
#         )

#         # Add risk registry data
#         risk_registry_data = [
#             {
#                 "risk_id": 1,
#                 "project_id": 101,
#                 "description": "Legacy system interdependencies may be undocumented, leading to unexpected issues during migration",
#                 "probability": "High",
#                 "impact": "High",
#                 "mitigation_strategy": "Conduct comprehensive dependency mapping before migration and implement extensive testing in staging environment",
#                 "owner": "Jennifer Adams",
#                 "status": "Active",
#             },
#             {
#                 "risk_id": 2,
#                 "project_id": 101,
#                 "description": "Security vulnerabilities during transition period",
#                 "probability": "Medium",
#                 "impact": "Critical",
#                 "mitigation_strategy": "Implement additional security monitoring during migration and conduct penetration testing before each phase goes live",
#                 "owner": "David Zhang",
#                 "status": "Active",
#             },
#             {
#                 "risk_id": 3,
#                 "project_id": 102,
#                 "description": "Changes to healthcare regulations during project implementation",
#                 "probability": "Medium",
#                 "impact": "High",
#                 "mitigation_strategy": "Maintain contact with regulatory experts and build flexibility into the system to accommodate regulatory changes",
#                 "owner": "Robert Chen",
#                 "status": "Active",
#             },
#             {
#                 "risk_id": 4,
#                 "project_id": 102,
#                 "description": "Integration challenges with legacy healthcare systems",
#                 "probability": "High",
#                 "impact": "High",
#                 "mitigation_strategy": "Develop comprehensive adapters and conduct early integration testing with actual legacy systems",
#                 "owner": "Robert Chen",
#                 "status": "Active",
#             },
#             {
#                 "risk_id": 5,
#                 "project_id": 103,
#                 "description": "AI model accuracy falls below required threshold",
#                 "probability": "Medium",
#                 "impact": "Critical",
#                 "mitigation_strategy": "Implement continuous model training and monitoring, with fallback to rule-based detection if accuracy drops",
#                 "owner": "Michael Patel",
#                 "status": "Active",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following risk data to the RiskRegistry table: {json.dumps(risk_registry_data)}"
#         )

#         # 7. Create ProductCatalog table
#         cls.knowledge_manager.store(
#             "Create a table called 'ProductCatalog' with columns for product_id (integer), "
#             "name (string), category (string), description (text), version (string), "
#             "release_date (date), price_tier (string), and features (string)."
#         )

#         # Add product catalog data
#         product_catalog_data = [
#             {
#                 "product_id": 1,
#                 "name": "CloudMigrate Pro",
#                 "category": "Infrastructure",
#                 "description": "Enterprise-grade cloud migration solution with automated discovery, planning, and execution capabilities",
#                 "version": "3.5.2",
#                 "release_date": "2023-01-15",
#                 "price_tier": "Enterprise",
#                 "features": "Automated dependency mapping, Zero-downtime migration, Multi-cloud support, Compliance verification",
#             },
#             {
#                 "product_id": 2,
#                 "name": "HealthRecord Plus",
#                 "category": "Healthcare",
#                 "description": "Comprehensive electronic health record system designed for multi-facility healthcare providers",
#                 "version": "2.8.0",
#                 "release_date": "2022-11-30",
#                 "price_tier": "Premium",
#                 "features": "HIPAA compliance, HL7 FHIR support, Telehealth integration, Advanced analytics",
#             },
#             {
#                 "product_id": 3,
#                 "name": "FinancialGuardian",
#                 "category": "Finance",
#                 "description": "AI-powered fraud detection and prevention system for financial institutions",
#                 "version": "4.1.3",
#                 "release_date": "2023-03-10",
#                 "price_tier": "Enterprise",
#                 "features": "Real-time transaction monitoring, Machine learning models, Regulatory reporting, Case management",
#             },
#             {
#                 "product_id": 4,
#                 "name": "RetailConnect",
#                 "category": "Retail",
#                 "description": "Omnichannel retail management platform with integrated e-commerce and in-store capabilities",
#                 "version": "3.2.1",
#                 "release_date": "2022-09-22",
#                 "price_tier": "Standard",
#                 "features": "Inventory management, Order processing, Customer management, Analytics dashboard",
#             },
#             {
#                 "product_id": 5,
#                 "name": "EcoTrack",
#                 "category": "Environmental",
#                 "description": "Sustainability monitoring and reporting solution for environmental compliance",
#                 "version": "1.5.0",
#                 "release_date": "2023-02-28",
#                 "price_tier": "Standard",
#                 "features": "Carbon footprint calculation, Regulatory reporting, Sustainability metrics, Improvement recommendations",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following product data to the ProductCatalog table: {json.dumps(product_catalog_data)}"
#         )

#         # 8. Create CompanyPolicies table
#         cls.knowledge_manager.store(
#             "Create a table called 'CompanyPolicies' with columns for policy_id (integer), "
#             "title (string), category (string), description (text), effective_date (date), "
#             "last_revised (date), approved_by (string), and compliance_requirement (string)."
#         )

#         # Add company policies data
#         company_policies_data = [
#             {
#                 "policy_id": 1,
#                 "title": "Data Security Policy",
#                 "category": "Security",
#                 "description": "This policy outlines the standards and procedures for protecting company and client data, including encryption requirements, access controls, and security incident response.",
#                 "effective_date": "2022-01-01",
#                 "last_revised": "2023-06-15",
#                 "approved_by": "Executive Board",
#                 "compliance_requirement": "ISO 27001, GDPR, CCPA",
#             },
#             {
#                 "policy_id": 2,
#                 "title": "Remote Work Policy",
#                 "category": "Human Resources",
#                 "description": "Guidelines for remote work arrangements, including equipment requirements, work hours, availability expectations, and security practices for remote employees.",
#                 "effective_date": "2020-03-15",
#                 "last_revised": "2023-02-10",
#                 "approved_by": "HR Department",
#                 "compliance_requirement": "None",
#             },
#             {
#                 "policy_id": 3,
#                 "title": "Client Confidentiality Agreement",
#                 "category": "Legal",
#                 "description": "Standard agreement governing the protection of client confidential information, intellectual property, and trade secrets encountered during project work.",
#                 "effective_date": "2018-05-20",
#                 "last_revised": "2022-11-30",
#                 "approved_by": "Legal Department",
#                 "compliance_requirement": "NDA Standards",
#             },
#             {
#                 "policy_id": 4,
#                 "title": "Project Management Methodology",
#                 "category": "Operations",
#                 "description": "Standardized approach to project management, including initiation, planning, execution, monitoring, and closure phases with defined deliverables and approval gates.",
#                 "effective_date": "2019-08-01",
#                 "last_revised": "2023-01-20",
#                 "approved_by": "Operations Director",
#                 "compliance_requirement": "PMI Standards",
#             },
#             {
#                 "policy_id": 5,
#                 "title": "Healthcare Data Handling Procedures",
#                 "category": "Security",
#                 "description": "Specific procedures for handling protected health information (PHI) in compliance with HIPAA and other healthcare regulations.",
#                 "effective_date": "2020-06-15",
#                 "last_revised": "2023-05-10",
#                 "approved_by": "Compliance Officer",
#                 "compliance_requirement": "HIPAA, HITECH Act",
#             },
#         ]
#         cls.knowledge_manager.store(
#             f"Add the following policy data to the CompanyPolicies table: {json.dumps(company_policies_data)}"
#         )

#         # Create derived columns and vector embeddings for search
#         cls.knowledge_manager.store(
#             "Create a derived column 'full_description' in the Projects table using the equation 'name + \" - \" + description'"
#         )

#         cls.knowledge_manager.store(
#             "Create a derived column 'client_project' in the Projects table using the equation 'f\"Client {client_id}: {name}\"'"
#         )

#     def test_complex_information_retrieval(self):
#         """Test the ability to retrieve complex, nuanced information spanning multiple tables."""

#         # Test retrieving information that requires joining data from multiple tables
#         result = self.knowledge_manager.retrieve(
#             "Find all critical priority requirements for healthcare projects and list the team members working on those projects"
#         )

#         # Verify response contains relevant information about healthcare requirements and team members
#         assert (
#             "HIPAA Compliance" in result
#         ), "Response should include HIPAA compliance requirement"
#         assert (
#             "Patient Management System" in result
#         ), "Response should mention Patient Management System"
#         assert (
#             "Robert Chen" in result
#         ), "Response should mention the project manager Robert Chen"

#         # Test retrieving information requiring complex filtering and aggregation
#         result = self.knowledge_manager.retrieve(
#             "What's the total budget for all active projects in the Technology industry, and what percentage of our overall active project budget does this represent?"
#         )

#         # Verify budget calculations are included in the response
#         assert (
#             "$450,000" in result or "450000" in result
#         ), "Response should include Technology industry project budget"
#         assert "%" in result, "Response should include percentage calculation"

#         # Test retrieving information requiring semantic understanding
#         result = self.knowledge_manager.retrieve(
#             "Which projects have the highest security concerns based on the risk registry and client requirements?"
#         )

#         # Verify response identifies high-security projects
#         assert (
#             "Cloud Migration" in result or "TechNova" in result
#         ), "Response should identify cloud migration security risks"
#         assert (
#             "security" in result.lower() and "risk" in result.lower()
#         ), "Response should discuss security risks"

#     def test_complex_knowledge_updates(self):
#         """Test the ability to update knowledge with complex changes that affect multiple tables."""

#         # Add a new project with related data across multiple tables
#         self.knowledge_manager.store(
#             "Add a new project to the Projects table with project_id 107, client_id 3 (Global Financial Services), name 'Investment Portfolio Management', "
#             "description 'Developing a comprehensive investment portfolio tracking and optimization system with regulatory compliance features', "
#             "start_date '2023-11-01', end_date '2024-07-31', budget 410000.00, status 'Planning', project_manager 'Samantha Brooks', "
#             "priority 'High', domain 'Investment Management'"
#         )

#         # Add related project phases
#         self.knowledge_manager.store(
#             "Add the following phases to the ProjectPhases table: "
#             "1) phase_id 1009, project_id 107, phase_name 'Requirements Analysis', start_date '2023-11-01', end_date '2023-12-15', "
#             "deliverables 'Requirements document, compliance checklist', status 'Not Started', resources_allocated 70000.00; "
#             "2) phase_id 1010, project_id 107, phase_name 'Design Phase', start_date '2023-12-16', end_date '2024-02-28', "
#             "deliverables 'Technical specifications, UI/UX designs', status 'Not Started', resources_allocated 90000.00"
#         )

#         # Add related client requirements
#         self.knowledge_manager.store(
#             "Add a new requirement to the ClientRequirements table with requirement_id 7, project_id 107, "
#             "title 'SEC Rule 606 Compliance', description 'The system must generate reports compliant with SEC Rule 606 for quarterly reporting of routing information', "
#             "priority 'Critical', status 'Planned', requested_by 'Lisa Williams', compliance_related true"
#         )

#         # Test retrieving the updated information
#         result = self.knowledge_manager.retrieve(
#             "What is the latest project added for Global Financial Services, and what are its phases and requirements?"
#         )

#         # Verify the updated information is correctly retrieved
#         assert (
#             "Investment Portfolio Management" in result
#         ), "Response should include the new project name"
#         assert "Requirements Analysis" in result
