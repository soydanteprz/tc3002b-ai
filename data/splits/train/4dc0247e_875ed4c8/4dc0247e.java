import java.io.*;
import java.util.*;


public class ComdeFormces {
	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		FastReader sc=new FastReader();
		BufferedWriter log = new BufferedWriter(new OutputStreamWriter(System.out));
		int t=sc.nextInt();
		while(t--!=0) {
			int n=sc.nextInt();
//			int m=sc.nextInt();
		    ArrayList<ArrayList<pair>> ar=new ArrayList<>();
		    ArrayList<pair> arr=new ArrayList<>();
		    int c[]=new int[n];
		    for(int i=0;i<=n;i++) {
		    	ar.add(new ArrayList<>());
		    }
		    for(int i=0;i<n-1;i++) {
		    	int a=sc.nextInt();
		    	int b=sc.nextInt();
		    	ar.get(a).add(new pair(b,i));
		    	ar.get(b).add(new pair(a,i));
		    	
		    }
		    boolean ans=true;
		    for(int i=0;i<=n;i++) {
		     if(ar.get(i).size()>2) {
		    	 ans=false;
		    	 break;
		     }  	
		    }
		    if(ans) {
		    
		       for(int i=1;i<=n;i++) {
		    	   if(ar.get(i).size()==1) {
		    		   boolean vis[]=new boolean[n+1];
		    		   dfs(ar,2,i,vis,c);
		    		   break;
		    	   }
		       }
		       for(int i=0;i<n;i++) {
		    	   if(c[i]!=0)log.write(c[i]+" ");
		       }
		    }
		    else log.write("-1");
		    
		     log.write("\n");
		     log.flush();
			}
		}
	
		
static void dfs(ArrayList<ArrayList<pair>>ar ,int val,int src,boolean vis[],int c[]) {
	if(vis[src])return;
	vis[src]=true;
	for(int i=0;i<ar.get(src).size();i++) {
		if(!vis[ar.get(src).get(i).a]) {
			c[ar.get(src).get(i).b]=val;
			if(val==2) {
				dfs(ar,3,ar.get(src).get(i).a,vis,c);
			}
			else {
				dfs(ar,2,ar.get(src).get(i).a,vis,c);
			}
		}
	}
}	
	
	
	
		

		
		


	
	
		
	

	
	
	
	
	
	
	
	
	
	
//utils
 
 
	
	
	
	
public static class pair{
	int a;
	int b;
	public pair(int a,int b) {
		this.a=a;
		this.b=b;
	}
	
	
}	


}
