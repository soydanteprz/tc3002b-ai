import java.util.*;
import java.io.*;

public class hmm {

	static Scanner sc = new Scanner(System.in);
	static PrintWriter pw = new PrintWriter(System.out);
	 static boolean visited[];
	 static ArrayList<pair>arr[];
	 static HashMap<String,Integer>hm = new HashMap<>();
   public static void dfs(int start, int prime) {
	   visited[start]=true;
	   for(pair x: arr[start]) {
		   if(!visited[(int) x.x]) {
			   x.y = prime==2?3:2;
			   hm.put(start+","+x.x,x.y);
			   hm.put(x.x+","+start, x.y);
			   dfs(x.x,x.y);
		   }
	   }
	   
   }
   

	public static void main(String[] args) throws Exception {
		int t =sc.nextInt();
		while (t-- > 0) {
		         int n =sc.nextInt();
		         visited= new boolean [n];
		         arr=new ArrayList[n];
		         int[]color = new int[n];
		         ArrayList<pair>lol = new ArrayList<pair>();
		         
		         for(int i=0;i<n;i++)
		        	 arr[i]=new ArrayList<pair>();
		         for(int i=0;i<n-1;i++) {
		        	 int u = sc.nextInt()-1;
		        	 int v = sc.nextInt()-1;
		             lol.add(new pair(u,v));
		        	 arr[u].add(new pair(v,0));
		        	 arr[v].add(new pair(u,0));
		         }
		         boolean can = true;
		         for(int i=0;i<n;i++)
		        	 if(arr[i].size()>2){
		        		 can = false;
		        	 }
		         if(!can)
		        	 pw.println(-1);
		         
		         else {
		        	 int []hh = new int [] {2,3};
		        	 int i=0;
		        	 visited[0]= true;
		        	 for(pair x: arr[0]) {
		        		 hm.put(0+","+x.x,hh[i]);
		        		 dfs(x.x,hh[i++]);
		        	 }
		        	 //dfs(0,3);
		        	 for(pair a:lol) {
		        		 int u = a.x;
		        		 int y = a.y;
		        		 if(hm.containsKey(u+","+y))
		        				 pw.print(hm.get(u+","+y)+" ");
		        		 else
		        			 pw.print(hm.get(y+","+u)+" ");
		        	 }
		        	 pw.println();
		         }
			
		}

		pw.close();

	}

	// -------------- stuff ------------------------------
	static class pair {
		int x ;
		int y;
	
		
		public pair(int n,int c) {
			x= n;
			y = c;
		
			
		}
	
	
		
	}

	

}
