import java.io.*; 
import java.util.*;

public class one
{
  static Scanner sc=new Scanner(System.in);
  boolean prime[];
  static int prev=-1;
  static int dp[][];

  

  public static void main(String[] args) {
    //int testcase=1;
    int testcase=sc.nextInt();
    //System.out.println("HI");
      while(testcase-->0){
        // int x=sc.nextInt();
        // int y=sc.nextInt();
       //String str[]=new String[size];
       solve();
       System.out.println();
    }    
  } 

  public static void solve(){
    HashMap<Integer,Integer> map=new HashMap<Integer,Integer>();
    int size=sc.nextInt();
    int arr[][]=new int[size-1][2];
    for(int i=0;i<size-1;i++){
      arr[i][0]=sc.nextInt();
      arr[i][1]=sc.nextInt();
    }
    for(int x[]:arr){
      map.put(x[0],map.getOrDefault(x[0], 0)+1);
      map.put(x[1],map.getOrDefault(x[1], 0)+1);
      if(map.get(x[0])>2||map.get(x[1])>2){
        System.out.println(-1);
        return;
      }
    }
    List<List<Integer>> adj=new ArrayList<>();
    for(int i=0;i<=size;i++)
    adj.add(new ArrayList<Integer>());
    for(int x[]:arr){
      adj.get(x[0]).add(x[1]);
      adj.get(x[1]).add(x[0]);
    }
    //System.out.println(adj);
    int vist[]=new int[size+1];
    HashMap<String,Integer> ans=new HashMap<String,Integer>();
    for(int i=1;i<=size;i++){
      if(vist[i]==0){
        dfs(i,vist,adj,ans,2);
      }
    }
    //System.out.println(ans);
    for(int x[]:arr){
      //System.out.print(map.get(x[0]));
      int a=Math.min(x[0],x[1]);
      int b=Math.max(x[0],x[1]);
      String s=a+" "+b;
      System.out.print(ans.get(s)+" ");
    }
    // map=new HashMap<Integer,Integer>();
    // for(int x[]:arr){
    //   if(map.containsKey(x[0])){
    //     int val=13-map.get(x[0]);
    //     map.put(x[1],val);
    //     System.out.print(val+" ");
    //   }else if(map.containsKey(x[1])){
    //     int val=13-map.get(x[1]);
    //     map.put(x[0],val);
    //     System.out.print(val+" ");
    //   }else{
    //     System.out.print(2+" ");
    //     map.put(x[0],2);
    //     map.put(x[1],2);
    //   }
    // }

  }
  public static void dfs(int node,int vist[],List<List<Integer>> adj,HashMap<String,Integer> ans,int val){
    vist[node]=1;
    for(int i:adj.get(node)){
      if(vist[i]==1)
      continue;
      int x=Math.min(i, node);
      int y=Math.max(i, node);
      ans.put(x+" "+y,val);
      dfs(i,vist,adj,ans,5-val);
      val=5-val;
    }
  }
}
