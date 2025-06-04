import java.util.*;
public class D{
  static Scanner sc;
  public static void solve(){
    int n=sc.nextInt();
    Integer a[]=new Integer[n];
    int flag;
    for(int i=0;i<n;i++) a[i]=sc.nextInt();
    String s=sc.next();
    ArrayList<Integer> x=new ArrayList<>();
    ArrayList<Integer> y=new ArrayList<>();
    for(int i=0;i<n;i++){
      if(s.charAt(i)=='B') x.add(a[i]);
      else y.add(a[i]);
    }
    Collections.sort(x);
    Collections.sort(y);
    int p=n;
    int q=1;
    for(int i=y.size()-1;i>=0;i--){
      if(y.get(i)>p){System.out.println("NO"); return;}
      p-=1;
    }
    for(int i=0;i<x.size();i++){
      if(x.get(i)<q){System.out.println("NO"); return;}
      q+=1;
    }
    System.out.println("YES");
  }
  public static void main(String args[]){
      sc=new Scanner(System.in);
      int t=sc.nextInt();
      while(t-->0) solve();
  }
}
