import java.io.*;




import java.util.*;
/*
 


 */





 
 
 
 
 public class C {
    static FastReader sc=null;
   
	public static void main(String[] args) {
		sc=new FastReader();
		int t=sc.nextInt();
		while(t-->0) {
			int n=sc.nextInt();
			Node nodes[]=new Node[n];
			for(int i=0;i<n;i++)nodes[i]=new Node(i,sc.nextInt(),sc.nextInt());
			for(int i=0;i+1<n;i++) {
				int v=sc.nextInt()-1,w=sc.nextInt()-1;
				nodes[v].adj.add(nodes[w]);
				nodes[w].adj.add(nodes[v]);
			}
			
			dfs(nodes[0],null);
			System.out.println(Math.max(nodes[0].right,nodes[0].left));
		}
		
		
		
	}
	static void dfs(Node v,Node par) {
		for(Node nn:v.adj) {
			if(nn!=par) {
				dfs(nn,v);
				v.right+=Math.max(nn.right+Math.abs(v.r-nn.r), 
						nn.left+Math.abs(v.r-nn.l));
				v.left+=Math.max(nn.right+Math.abs(v.l-nn.r), 
						nn.left+Math.abs(v.l-nn.l));
			}
		}
	}
	static class Node{
		int id;
		long right=0,left=0;
		int l,r;
		ArrayList<Node> adj=new ArrayList<>();
		
		Node(int id,int l,int r){
			this.id=id;
			this.l=l;
			this.r=r;
		}
	}
	
	
	
    
    
    
    
    
    
   
 
      
      
        
 
    
    
    
    
    
    
    
    
    
    
     
}
