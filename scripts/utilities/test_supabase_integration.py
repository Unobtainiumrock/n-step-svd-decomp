#!/usr/bin/env python3
"""
Test Supabase Integration for Post Content Lookup
"""

import os
import json

def test_supabase_connection():
    """Test the Supabase connection and table access."""
    
    print("🔗 TESTING SUPABASE INTEGRATION")
    print("=" * 50)
    
    # Check environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    
    print(f"Environment Variables:")
    print(f"  ├─ SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Missing'}")
    print(f"  └─ SUPABASE_ANON_KEY: {'✅ Set' if supabase_key else '❌ Missing'}")
    
    if not (supabase_url and supabase_key):
        print(f"\n❌ Missing Supabase credentials!")
        print(f"Add to your .env file:")
        print(f"SUPABASE_URL=https://your-project.supabase.co")
        print(f"SUPABASE_ANON_KEY=your_anon_key_here")
        return
    
    try:
        from supabase import create_client
        
        # Test connection
        supabase = create_client(supabase_url, supabase_key)
        print(f"\n✅ Supabase client created successfully")
        
        # Test table access
        print(f"\n🔍 Testing table access...")
        
        # Test ugc_posts table
        try:
            response = supabase.table('ugc_posts').select('id, post_text').limit(1).execute()
            if response.data:
                print(f"  ├─ ugc_posts table: ✅ Accessible")
                sample_post = response.data[0]
                print(f"  │   └─ Sample ID: {sample_post.get('id', 'N/A')}")
                text_preview = sample_post.get('post_text', '')[:50] + '...' if sample_post.get('post_text') else 'No text'
                print(f"  │   └─ Sample text: {text_preview}")
            else:
                print(f"  ├─ ugc_posts table: ⚠️  Empty or no access")
        except Exception as e:
            print(f"  ├─ ugc_posts table: ❌ Error - {e}")
        
        # Test ugc_videos table
        try:
            response = supabase.table('ugc_videos').select('id, script').limit(1).execute()
            if response.data:
                print(f"  └─ ugc_videos table: ✅ Accessible")
                sample_video = response.data[0]
                print(f"      └─ Sample ID: {sample_video.get('id', 'N/A')}")
                script_preview = sample_video.get('script', '')[:50] + '...' if sample_video.get('script') else 'No script'
                print(f"      └─ Sample script: {script_preview}")
            else:
                print(f"  └─ ugc_videos table: ⚠️  Empty or no access")
        except Exception as e:
            print(f"  └─ ugc_videos table: ❌ Error - {e}")
            
        print(f"\n✅ Supabase integration test complete!")
        
    except ImportError:
        print(f"\n❌ Supabase library not installed")
        print(f"Install with: pip install supabase")
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")

def test_id_routing():
    """Test the ID routing logic."""
    
    print(f"\n🔄 TESTING ID ROUTING LOGIC")
    print(f"=" * 40)
    
    # Test cases
    test_cases = [
        ("12345", "ugc_posts", "12345"),
        ("67890_video", "ugc_videos", "67890"),
        ("123_video", "ugc_videos", "123"),
        ("999999", "ugc_posts", "999999")
    ]
    
    for pinecone_id, expected_table, expected_id in test_cases:
        if pinecone_id.endswith('_video'):
            clean_id = pinecone_id.replace('_video', '')
            table = 'ugc_videos'
        else:
            clean_id = pinecone_id
            table = 'ugc_posts'
        
        status = "✅" if (table == expected_table and clean_id == expected_id) else "❌"
        print(f"  {status} {pinecone_id} → table={table}, id={clean_id}")

if __name__ == "__main__":
    test_supabase_connection()
    test_id_routing()
