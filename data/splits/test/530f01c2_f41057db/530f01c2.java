import java.util.*;
import java.lang.*;
import java.io.*;

/* Name of the class has to be "Main" only if the class is public. */
public class Codechef
{
	public static void main (String[] args) throws java.lang.Exception
	{
		Scanner sc = new Scanner(System.in);
		PrintWriter out = new PrintWriter(System.out);
		int T = sc.nextInt();
		for(int t = 0; t<T; t++){
		    int n = sc.nextInt(), aS = sc.nextInt() - 1, bS = sc.nextInt() - 1, aDis = sc.nextInt(), bDis = sc.nextInt();
		    Node nodes[] = new Node[n];
		    for(int i = 0; i<n; i++)nodes[i] = new Node();
		    for(int i = 0; i<n - 1; i++){
		        int g = sc.nextInt() - 1, h = sc.nextInt() - 1;
		        nodes[g].adj.add(nodes[h]); nodes[h].adj.add(nodes[g]);
		    }
		    if(aDis * 2 >= bDis){
		        out.println("Alice"); continue;
		    }
		    BFS(nodes, nodes[bS]);
		    if (nodes[aS].dis<=aDis) {
				//alice wins in one move
				out.println("Alice");
				continue;
			}
			Node farthest=farthest(nodes);
			BFS(nodes, farthest);
			farthest=farthest(nodes);
			int farDist = farthest.dis;
		    if(farDist <= aDis * 2)out.println("Alice");
		    else out.println("Bob");
		}
		out.flush();
	}
	static Node farthest(Node[] nodes){
	    Node ans = nodes[0];
	    for(Node nn: nodes){
	        if(nn.dis > ans.dis)ans = nn;
	    }
	    return ans;
	}
	static void BFS(Node nodes[], Node prev){
	    for (Node nn:nodes) nn.dis=-1;
	    Queue<Node> queue = new LinkedList();
	    queue.add(prev);
	    prev.dis = 0;
	    while(!queue.isEmpty()){
	        Node par = queue.poll();
	        for(Node c: par.adj){
	            if(c.dis == -1){
	                c.dis = par.dis + 1;
	                queue.add(c);
	            }
	        }
	    }
	}
	static class Node{
	    List<Node> adj = new ArrayList();
	    int dis = -1;
	}
}
