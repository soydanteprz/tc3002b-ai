import java.io.*;

import java.util.*;
 public class Solution {
	 
	static long res;
	public static void main(String[] args) throws Exception
    {
		FastReader fr=new FastReader();
        int n=fr.nextInt();
        ArrayList<Integer> oc=new ArrayList<>();
        ArrayList<Integer> em=new ArrayList<>();
        res=Long.MAX_VALUE;
        for(int i=0;i<n;i++) {
        	int v=fr.nextInt();
        	if(v==1)
        		oc.add(i);
        	else
        		em.add(i);
        }
        Collections.sort(oc);
        Collections.sort(em);
        long dp[][]=new long[5001][5001];
        for(int i=0;i<dp.length;i++) {
        	for(int j=0;j<dp[i].length;j++) {
        		dp[i][j]=-1;
        	}
        }
        System.out.println(getMin(oc,em,0,0,dp));
        
        
		
    }
	public static long getMin(ArrayList<Integer> oc,ArrayList<Integer> em,int idx,int j,long dp[][]) {
	   if(idx==oc.size())
		   return 0;
		long available=em.size()-j;
		long req=oc.size()-idx;
		if(available<req)
			return Integer.MAX_VALUE;
		if(dp[idx][j]!=-1)
			return dp[idx][j];
		
		long ch1=getMin(oc,em,idx,j+1,dp);
		long ch2=getMin(oc,em,idx+1,j+1,dp)+Math.abs(em.get(j)-oc.get(idx));
		
		return dp[idx][j]=Math.min(ch1,ch2);
		
	}
	
		
	
	 

	
	 
	 
	 
	 
	 
	 
	 
	
		
	 
	 
	 
	 
	 
	
	 
	 
	
	 
	 
            	  
             
             
    }


	
	 
	
	

	 
 
  


