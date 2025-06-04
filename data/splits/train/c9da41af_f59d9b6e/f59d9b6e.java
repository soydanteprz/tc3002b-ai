import java.util.*;
import java.lang.*;
import java.io.*;

/* Name of the class has to be "Main" only if the class is public. */
public class Codechef{
    public static class Edge{
        int node;
        int index;
        Edge(int node, int index){
            this.node = node;
            this.index = index;
        }
    }
    static Scanner scn = new Scanner(System.in);
	public static void main (String[] args) throws java.lang.Exception{
		int t = scn.nextInt();
		while(t-->0){
		    solve();
		}
	}
	public static void solve(){
	    int n = scn.nextInt();
	    ArrayList<Edge>[]graph = new ArrayList[n];
	    for(int i = 0; i < n; i++){
	        graph[i] = new ArrayList<>();
	    }
	    for(int i = 0; i < n - 1; i++){
	        int u = scn.nextInt() - 1;
	        int v = scn.nextInt() - 1;
	        graph[u].add(new Edge(v, i));
	        graph[v].add(new Edge(u, i));
	    }
	    int start = 0;
	    for(int i = 0; i < n; i++){
	        if(graph[i].size() > 2){
	            System.out.println("-1");
	            return;
	        }else if(graph[i].size() == 1){
	            start = i;
	        }
	    }
	    int[]weight = new int[n - 1];
	    int prevNode = -1, curNode = start, curWeight = 2;
	    while(true){
	        ArrayList<Edge>edges = graph[curNode];
	        Edge next = edges.get(0);
	        if(next.node == prevNode){
	            if(edges.size() == 1){
	                break;
	            }else{
	                next = edges.get(1);
	            }
	        }
	        weight[next.index] = curWeight;
	        prevNode = curNode;
	        curNode = next.node;
	        curWeight = 5 - curWeight;
	    }
	    for(int i = 0; i < n - 1; i++){
	        System.out.print(weight[i]);
	        System.out.print(" ");
	    }
	    System.out.println();
	}
}
