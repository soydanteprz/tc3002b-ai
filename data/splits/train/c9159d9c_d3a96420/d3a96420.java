import java.util.*;

public class Main {
  public static void main(String[] args) {
    Scanner sc= new Scanner(System.in);
    int t=sc.nextInt();
    while(t-->0){
      int n=sc.nextInt();
      int[] a=new int[n];
      for(int i=0;i<n;i++)a[i]=sc.nextInt();
      String x=sc.next();
      Vector<Integer> R=new Vector<>();
      Vector<Integer> B=new Vector<>();
      for(int i=0;i<n;i++){
        if(x.charAt(i)=='B') R.add(a[i]);
        else B.add(a[i]);
      }
      Collections.sort(R);
      Collections.sort(B);
      boolean yes=true;
      for(int i=0;i<R.size();i++){
        if(R.get(i)-i<1){System.out.println("NO");yes=false;break;}
      }
      if(yes)
        {
          int s=B.size();
          for(int j=0;j<s;j++){
            if(B.get(j)+s-j>n+1){System.out.println("NO");yes=false;break;}
          }
        }
        if(yes)System.out.println("YES");
    }
    sc.close();
  }
}
