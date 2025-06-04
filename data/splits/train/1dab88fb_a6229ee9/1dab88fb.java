import java.util.*;

public class Main
{
    static class Edge{
        public int node;
        public int index;
        public Edge(int n, int i){
            node=n;
            index=i;
        }
    }
    static Scanner sc=new Scanner(System.in);
	public static void main(String[] args) {
	    
	    int test=sc.nextInt();
	    while(test-->0){
	        
	        solve();
	    }
	}
	
	static void solve(){
	    int n=sc.nextInt();
	        ArrayList<ArrayList<Edge>> graph= new ArrayList<ArrayList<Edge>>();
	        for(int i=0;i<n;i++){
	            graph.add(new ArrayList<>());
	        }
	        for (int i = 0; i < n - 1; i++) {
	            int u = sc.nextInt();
	            int v = sc.nextInt();
	            u--; v--;
	            
	            graph.get(u).add(new Edge(v, i));
	            graph.get(v).add(new Edge(u, i));
            }
            
            int start = 0;
            for (int i = 0; i < n; i++) {
                if (graph.get(i).size() > 2) {
                    System.out.println("-1");
                    return;
                } else if (graph.get(i).size() == 1) {
                    start = i;
                }
            }
            int[] weight = new int[n - 1];
            
            int prevNode = -1;
            int curNode = start;
            int curWeight = 2;
            
            while (true) {
                ArrayList<Edge> edges = graph.get(curNode);
                Edge next = edges.get(0);
                
                if (next.node == prevNode) {
                    if (edges.size() == 1) {
                        break;
                    } else {
                        next = edges.get(1);
                    }
                }
                weight[next.index] = curWeight;
                
                prevNode = curNode;
                curNode = next.node;
                curWeight = 5 - curWeight;
            }
            for (int i = 0; i < n - 1; i++) {
                System.out.print(weight[i]);
                System.out.print(" ");
            }
            System.out.println();
	}
}
