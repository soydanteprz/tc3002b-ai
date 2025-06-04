import java.io.*;
import java.util.*;

public class D669b{
   
   public static void main(String[] args)throws IOException{
      BufferedReader f = new BufferedReader(new InputStreamReader(System.in));
      PrintWriter out = new PrintWriter(System.out);
      
      int n = Integer.parseInt(f.readLine());
      
      StringTokenizer st = new StringTokenizer(f.readLine());
      
      int[] array = new int[n];
      
      for(int k = 0; k < n; k++){
         array[k] = Integer.parseInt(st.nextToken());
      }
      
      ArrayList<HashSet<Integer>> adj = new ArrayList<HashSet<Integer>>(n);
      for(int k = 0; k < n; k++) adj.add(new HashSet<Integer>());
      
      for(int k = 0; k < n-1; k++){
         adj.get(k).add(k+1);
      }
      
      //closest number before that is <
      Stack<Num> stk = new Stack<Num>();
      stk.add(new Num(array[0],0));
      
      for(int k = 1; k < n; k++){
         while(!stk.isEmpty() && stk.peek().x > array[k]){
            stk.pop();
         }
         
         if(!stk.isEmpty()){
            adj.get(stk.peek().i).add(k);
         }
         
         stk.add(new Num(array[k],k));
      }
      
      //closest number after that is <
      stk = new Stack<Num>();
      stk.add(new Num(array[n-1],n-1));
      
      for(int k = n-2; k >= 0; k--){
         while(!stk.isEmpty() && stk.peek().x > array[k]){
            stk.pop();
         }
         
         if(!stk.isEmpty()){
            adj.get(k).add(stk.peek().i);
         }
         
         stk.add(new Num(array[k],k));
      }
      //closest number before that is >
      stk = new Stack<Num>();
      stk.add(new Num(array[0],0));
      
      for(int k = 1; k < n; k++){
         while(!stk.isEmpty() && stk.peek().x < array[k]){
            stk.pop();
         }
         
         if(!stk.isEmpty()){
            adj.get(stk.peek().i).add(k);
         }
         
         stk.add(new Num(array[k],k));
      }
      
      //closest number after that is >
      stk = new Stack<Num>();
      stk.add(new Num(array[n-1],n-1));
      
      for(int k = n-2; k >= 0; k--){
         while(!stk.isEmpty() && stk.peek().x < array[k]){
            stk.pop();
         }
         
         if(!stk.isEmpty()){
            adj.get(k).add(stk.peek().i);
         }
         
         stk.add(new Num(array[k],k));
      }
      
      
      int[] path = new int[n];
      Arrays.fill(path,Integer.MAX_VALUE);
      path[0] = 0;
      
      for(int k = 0; k < n; k++){
         for(int nei : adj.get(k)){
            path[nei] = Math.min(path[nei],path[k]+1);
         }
      }
      
      int answer = path[n-1];
      out.println(answer);
         
      
      
      
      out.close();
   }
   
   public static class Num{
      int x;
      int i;
      public Num(int a, int b){
         x = a;
         i = b;
      }
   }
      
}
