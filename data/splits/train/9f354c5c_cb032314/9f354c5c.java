import java.util.*;

import java.io.*;
import java.math.*;
import java.sql.Array;
import java.sql.ResultSet;
import java.sql.SQLException;



public class Solution {
	 
	  
	 
	  static long mod = (long)(1e9 + 7);
	 
	
	
	
	

	
	
	 
	 

	 
	   
	   
	 
	   
     
	      
	      
	      
	      
	      
	      
	      
	      
/************************************************ Query **************************************************************************************/	  
	 
/***************************************** 		Sparse Table	********************************************************/
	      
	      
	      
/********************************************************	Segement Tree	*****************************************************/
/**
	 	 static class SegmentTree{
			 long[] tree;
			 long[] arr;
			 int n;
			 SegmentTree(long[] arr){
				 this.n = arr.length;
				 tree = new long[4*n+1];
				 this.arr = arr;
				 buildTree(0, n-1, 1);
			 }
			 
			 
			  void buildTree(int s ,int e  ,int index ) {
					if(s == e) {
						tree[index] = arr[s];
						return;
					}
				
					int mid = (s+e)/2;
					
					buildTree( s,  mid, 2*index);
					buildTree( mid+1, e, 2*index+1);
					
					tree[index] = Math.min(tree[2*index] , tree[2*index+1]);
				}
			  
			 long query(int si ,int ei) {
				 return query(0 ,n-1 , si ,ei , 1   );
			 }
			 private long query( int ss ,int se ,int qs , int qe,int index) {
					
					if(ss>=qs && se<=qe) return tree[index];
					
					if(qe<ss || se<qs) return (long)(1e17);
					
					int mid = (ss + se)/2;
					long left = query( ss , mid , qs ,qe , 2*index);
					long right= query(mid + 1 , se , qs ,qe , 2*index+1);
					return Math.min(left, right);
				}
			 public void update(int index , int val) {
				 arr[index] = val;
				 for(long e:arr) System.out.print(e+" ");
				 update(index , 0 , n-1 , 1);
			 }
			 private void update(int id ,int si , int ei , int index) {
				 if(id < si || id>ei) return;
				 if(si == ei ) { 
					 tree[index] = arr[id];
					 return;
				 }
				 if(si > ei) return;
				 int mid = (ei + si)/2;
					
					update( id,  si, mid , 2*index);
					update( id , mid+1, ei , 2*index+1);
					
					tree[index] = Math.min(tree[2*index] ,tree[2*index+1]);
			 }
			  
		 }
		 */

/* ***************************************************************************************************************************************************/	 
	 
//	      static MyScanner sc = new MyScanner(); // only in case of less memory
	      static Reader sc = new Reader();
	 static StringBuilder sb = new StringBuilder();
	 public static void main(String args[]) throws IOException {

		 int tc = 1;
		  tc = sc.nextInt();

		 for(int i = 1 ; i<=tc ; i++) {

//			 sb.append("Case #" + i + ": "  );	// During KickStart && HackerCup
			 TEST_CASE();
			  
		 }
		 System.out.println(sb);
	 }

	 static void TEST_CASE() {
		int n = sc.nextInt();
		ArrayList<ArrayList<Integer>> adj = new ArrayList<>();
		for(int i =0 ; i<n ; i++) adj.add(new ArrayList<>());
		int[] U = new int[n-1] , V = new int[n-1];
		for(int i = 0 ; i<n-1 ; i++) {
			int u = sc.nextInt()-1 , v = sc.nextInt()-1;
			U[i] = u; V[i] = v;
			adj.get(u).add(v);
			adj.get(v).add(u);
		}
		int ind = -1;
		for(int i =0 ; i<n ; i++) {
			if(adj.get(i).size()>2) {
				sb.append("-1\n");
				return;
			}
			if(adj.get(i).size() == 1) {
				ind = i;
			}
		}
		Map<String , Integer> map = new HashMap<>();
		dfs(adj, ind, -1, true, map);
		for(int i =0 ; i<n-1 ; i++) {
		 int u = U[i];
		 int v = V[i];
		 if(map.containsKey(u+" "+v)) sb.append(map.get(u+" "+v)+" ");
		 else sb.append(map.get(v+" "+u)+" ");
		}
		sb.append("\n");
		
	 }
	 static void dfs(ArrayList<ArrayList<Integer>> adj ,int u , int p ,boolean eve ,Map<String , Integer> map ) {
		 if(eve) {
			 map.put(u+" "+p, 2);
		 }else {
			 map.put(u+" "+p, 3);
		 }
		 for(int v:adj.get(u)) {
			 if(v == p) continue;
			 eve = !eve;
			 dfs(adj, v, u, eve, map);
		 }
	 }
	}
