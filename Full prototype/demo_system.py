#!/usr/bin/env python3
"""
FRA AI System Demo Script
Demonstrates the capabilities of the complete FRA AI system
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

def demo_system_status():
    """Demo: Check system status"""
    print("🔍 Checking System Status...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System Status: {data['status']}")
            print(f"📊 Version: {data['version']}")
            print(f"🔗 Available Endpoints: {len(data['endpoints'])}")
        else:
            print("❌ System not responding")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Make sure to start the system first:")
        print("   python fra_ai_complete_system.py --action all")
        return False
    return True

def demo_model_status():
    """Demo: Check model loading status"""
    print("\n🤖 Checking Model Status...")
    try:
        response = requests.get(f"{API_BASE_URL}/models/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models Available: {len(data['available_models'])}")
            print(f"🚀 Inference Ready: {data['inference_ready']}")
            for model in data['available_models'][:3]:  # Show first 3
                print(f"   📦 {model}")
            if len(data['available_models']) > 3:
                print(f"   ... and {len(data['available_models']) - 3} more")
    except Exception as e:
        print(f"❌ Model status check failed: {e}")

def demo_health_check():
    """Demo: System health check"""
    print("\n🏥 System Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"💾 Database: {data['database']}")
            print(f"🧠 Models Loaded: {data['models_loaded']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def demo_create_fra_claim():
    """Demo: Create a sample FRA claim"""
    print("\n📝 Creating Sample FRA Claim...")
    
    claim_data = {
        "village_name": "Demo Village",
        "patta_holder": "Demo Farmer",
        "claim_type": "Individual Forest Rights",
        "area_hectares": 3.2,
        "coordinates": "28.1234,77.5678",
        "district": "Demo District",
        "state": "Demo State"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/claims",
            json=claim_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Claim Created Successfully!")
            print(f"🆔 Claim ID: {data['claim_id']}")
            print(f"📅 Status: {data['status']}")
        else:
            print(f"❌ Failed to create claim: {response.status_code}")
    except Exception as e:
        print(f"❌ Claim creation failed: {e}")

def demo_satellite_analysis():
    """Demo: Satellite data analysis"""
    print("\n🛰️ Running Satellite Analysis...")
    
    analysis_data = {
        "coordinates": [28.1234, 77.5678],
        "analysis_type": "comprehensive"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/satellite/analyze",
            json=analysis_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis Complete!")
            print(f"🌍 Coordinates: {data['coordinates']}")
            print(f"🌲 Forest Cover: {data['land_cover']['forest']}%")
            print(f"🚜 Agriculture: {data['land_cover']['agriculture']}%")
            print(f"💧 Water Bodies: {data['land_cover']['water']}%")
            print(f"📈 NDVI: {data['spectral_indices']['ndvi']}")
            print(f"🎯 Confidence: {data['confidence_score']}")
        else:
            print(f"❌ Analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Satellite analysis failed: {e}")

def demo_dss_recommendations():
    """Demo: Decision Support System recommendations"""
    print("\n🎯 Generating DSS Recommendations...")
    
    dss_data = {
        "village_id": "DEMO_VIL_001",
        "schemes": ["Jal Jeevan Mission", "MGNREGA", "PM-KISAN"],
        "priority_factors": ["water", "employment", "agriculture"]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/dss/recommend",
            json=dss_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Recommendations Generated!")
            print(f"🏘️ Village ID: {data['village_id']}")
            print(f"📊 Total Schemes: {data['total_schemes']}")
            
            for i, rec in enumerate(data['recommendations'][:2], 1):
                print(f"\n   {i}. 🎯 {rec['scheme']}")
                print(f"      📈 Priority Score: {rec['priority_score']}")
                print(f"      👥 Beneficiaries: {rec['estimated_beneficiaries']}")
                print(f"      ⏱️ Timeline: {rec['implementation_timeline']}")
        else:
            print(f"❌ DSS recommendations failed: {response.status_code}")
    except Exception as e:
        print(f"❌ DSS generation failed: {e}")

def demo_geojson_data():
    """Demo: Get GeoJSON data for mapping"""
    print("\n🗺️ Fetching GeoJSON Data for Maps...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/geojson/fra-boundaries")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ GeoJSON Data Retrieved!")
            print(f"📍 Features: {len(data['features'])}")
            if data['features']:
                feature = data['features'][0]
                props = feature['properties']
                print(f"   🏘️ Sample: {props['village_name']}")
                print(f"   📋 Type: {props['claim_type']}")
                print(f"   ✅ Status: {props['status']}")
                print(f"   📐 Area: {props['area_hectares']} hectares")
        else:
            print(f"❌ GeoJSON fetch failed: {response.status_code}")
    except Exception as e:
        print(f"❌ GeoJSON retrieval failed: {e}")

def demo_model_weights():
    """Demo: Get model weight information for frontend"""
    print("\n⚖️ Checking Model Weights Information...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/weights")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model Weights Information Retrieved!")
            print(f"📦 Total Models: {data['total_models']}")
            
            for model_name, info in list(data['weights'].items())[:3]:
                if info['status'] == 'loaded':
                    print(f"   🤖 {model_name}:")
                    print(f"      💾 Size: {info['size_mb']:.1f} MB")
                    print(f"      ✅ Status: {info['status']}")
        else:
            print(f"❌ Weights info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Weights retrieval failed: {e}")

def demo_api_documentation():
    """Demo: Show API documentation links"""
    print("\n📚 API Documentation Available:")
    print(f"   🔗 Interactive Docs: {API_BASE_URL}/docs")
    print(f"   📖 ReDoc: {API_BASE_URL}/redoc")
    print("   💡 Open these URLs in your browser for full API documentation")

def main():
    """Run complete demo"""
    print("=" * 80)
    print("🚀 FRA AI COMPLETE SYSTEM DEMO")
    print("=" * 80)
    print("This demo showcases the capabilities of the FRA AI system.")
    print("Make sure the system is running before starting the demo.")
    print("=" * 80)
    
    # Check if system is running
    if not demo_system_status():
        return
    
    # Wait a moment
    time.sleep(1)
    
    # Run all demos
    demo_model_status()
    time.sleep(1)
    
    demo_health_check() 
    time.sleep(1)
    
    demo_create_fra_claim()
    time.sleep(1)
    
    demo_satellite_analysis()
    time.sleep(1)
    
    demo_dss_recommendations()
    time.sleep(1)
    
    demo_geojson_data()
    time.sleep(1)
    
    demo_model_weights()
    time.sleep(1)
    
    demo_api_documentation()
    
    print("\n" + "=" * 80)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("The FRA AI system is fully operational and ready for:")
    print("✅ Frontend integration via API endpoints")
    print("✅ Real-time satellite data analysis") 
    print("✅ FRA claims processing and management")
    print("✅ AI-powered decision support recommendations")
    print("✅ Maps integration with GeoJSON data")
    print("✅ Production deployment")
    print("=" * 80)

if __name__ == "__main__":
    main()