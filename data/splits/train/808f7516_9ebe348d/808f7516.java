import java.util.*;
import java.io.*;
public class EdD {
	public static void main(String[] args) throws Exception{
		int num = 998244353;

		// TODO Auto-generated method stub
 		BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));
 		PrintWriter out = new PrintWriter(System.out);
//	 		String input1 = bf.readLine().trim();
//	 		String input2 = bf.readLine().trim();
 		int n = Integer.parseInt(bf.readLine());
 		int[] array = new int[n];
 		StringTokenizer st = new StringTokenizer(bf.readLine());
 		for(int j = 0;j<n;j++){
 			array[j] = Integer.parseInt(st.nextToken());
 		}
 		Map<Integer, Set<Integer>> mp = new HashMap<Integer, Set<Integer>>();
 		for(int j =0;j<n;j++)
 			mp.put(j, new HashSet<Integer>());
 		int[] nextge = new int[n];
 		int[] nextle = new int[n];
 		int[] prevge = new int[n];
 		int[] prevle = new int[n];
 		nextge[n-1] = -1;
 		nextle[n-1] = -1;
 		prevge[0] = -1;
 		prevle[0] = -1;
 		for(int j = n-2;j>=0;j--){
 			if (array[j+1] < array[j]){
 				nextle[j] = j+1;
 				mp.get(j).add(j+1);
 				int temp = j+1;
 				while(temp!= -1 && array[temp] < array[j])
 					temp = nextge[temp];
 				nextge[j] = temp;
 				if (temp!= -1)
 					mp.get(j).add(temp);
 			}
 			else if (array[j+1] > array[j]){
 				nextge[j] = j+1;
 				mp.get(j).add(j+1);
 				int temp = j+1;
 				while(temp!= -1 && array[temp] > array[j])
 					temp = nextle[temp];
 				nextle[j] = temp;
 				if (temp!= -1)
 					mp.get(j).add(temp);
 			}
 			else{
 				nextge[j] = j+1;
 				mp.get(j).add(j+1);
 				nextle[j] = j+1;
 				mp.get(j).add(j+1);
 			}	
 		}
 		for(int j = 1;j<n;j++){
 			if (array[j] < array[j-1]){
 			
 				prevge[j] = j-1;
 				mp.get(j-1).add(j);
 				int temp = j-1;
 				while(temp!= -1 && array[temp] > array[j])
 					temp = prevle[temp];
 				prevle[j] = temp;
 				if (temp!= -1)
 					mp.get(temp).add(j);
 			}
 			else if (array[j] > array[j-1]){
 				prevle[j] = j-1;
 				mp.get(j-1).add(j);
 				int temp = j-1;
 				while(temp!= -1 && array[temp] < array[j])
 					temp = prevge[temp];
 				prevge[j] = temp;
 				if (temp!= -1)
 					mp.get(temp).add(j);
 			}
 			else{
 				prevge[j] = j-1;
 				prevle[j] = j-1;
 				mp.get(j-1).add(j);
 			}	
 		}
	 	int[] depth = new int[n+1];
 		Set<Integer> seen = new HashSet<Integer>();
 		Queue<Integer> bfs = new LinkedList<Integer>();
 		bfs.add(0);
 		seen.add(0);
 		while(!bfs.isEmpty()){
 			int v = bfs.remove();
 			for(int child : mp.get(v)){
 				if (!seen.contains(child)){
 					bfs.add(child);
 					seen.add(child);
 					depth[child] = depth[v]+1;
 				}
 			}
 		}
 		
 		
 		out.println(depth[n-1]);
	 		
 		out.close();
 		
 		
 		
 	}
}
